import os
import wandb
import argparse
import datetime
import time

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import albumentations as A

from dataset import XRayDataset
from utils import set_seed, save_model, dice_coef


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]


def main(args):
    # Seed
    set_seed(args.seed)
    
    # Checkpoint directory
    save_dir_root = './work_dirs'
    if not os.path.isdir(save_dir_root) :
        os.mkdir(save_dir_root)
    
    save_dir = os.path.join(save_dir_root, args.save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # Model
    model = smp.UnetPlusPlus(
                encoder_name=args.encoder_name, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=args.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=29                      # model output channels (number of classes in your dataset)
            )
    wandb.watch(model)

    # Augmentations
    train_tf = A.Resize(1024, 1024)
    valid_tf = A.Resize(1024, 1024)
    
    # Dataset
    train_dataset = XRayDataset(
        image_root=args.image_root, label_root=args.label_root, is_train=True,
        transforms=train_tf, clahe=args.clahe, copypaste=args.cp,
        n_splits=args.n_splits, n_fold=args.n_fold)
    valid_dataset = XRayDataset(
        image_root=args.image_root, label_root=args.label_root, is_train=False,
        transforms=valid_tf, clahe=args.clahe, copypaste=False,
        n_splits=args.n_splits, n_fold=args.n_fold)

    # Dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    
    # Loss function
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    
    # Scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=50,
            cycle_mult=1.0,
            max_lr=args.lr,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.5
        )
    elif args.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(args.epochs*0.5)],
            last_epoch=args.epochs,
            verbose=True
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.8,
            patience=10,
            cooldown=10,
            eps=1e-6
        )
    
    
    # init best score
    best_dice = 0.
    
    # Training
    print(f'Start training..')
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        train_epoch_start = time.time()
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # logging with CLI
            if (step + 1) % (len(train_loader)//10) == 0:
                print(
                    f'Epoch [{epoch+1:3d}/{args.epochs}] | '
                    f'Step [{step+1:3d}/{len(train_loader)}] | '
                    f'Loss: {round(loss.item(), 4)}'
                )
            
                # logging with wandb
                wandb.log({
                    "Epoch":epoch+1,
                    "Train Loss":loss.item(),
                    "Learning Rate":optimizer.param_groups[0]["lr"]
                })
            
        mean_train_loss = total_train_loss/len(train_loader)
        print(f"Train Mean Loss : {round(mean_train_loss, 4)}")
        
        elapsed_train_time = datetime.timedelta(seconds=round(time.time() - train_epoch_start))
        print(f"Elapsed Training Time: {elapsed_train_time}")
        print(f"ETA : {elapsed_train_time * (args.epochs - epoch+1)}")
        
        if (epoch + 1) % args.val_every == 0:
            # Validation
            print(f'Start validation #{epoch+1:2d}')
            set_seed(args.seed)

            # init metrics
            total_valid_loss = 0
            dices = []
            
            model.eval()
            with torch.inference_mode():
                for step, (images, masks) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                    images, masks = images.cuda(), masks.cuda()
                    model = model.cuda()
                    
                    outputs = model(images)
                    
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    
                    # restore original size
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    loss = criterion(outputs, masks)
                    total_valid_loss += loss.item()
                    
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > args.valid_threshold).detach().cpu()
                    masks = masks.detach().cpu()
                    
                    dice = dice_coef(outputs, masks)
                    dices.append(dice)
                
                # mean validation loss
                mean_valid_loss = total_valid_loss/len(valid_loader)
                
                dices = torch.cat(dices, 0)
                dices_per_class = torch.mean(dices, 0)
                dice_str = [
                    f"{c:<12}: {d.item():.4f}"
                    for c, d in zip(CLASSES, dices_per_class)
                ]
                dice_str = "\n".join(dice_str)
                print(dice_str)
                print(f"Valid Mean Loss : {round(mean_valid_loss, 4)}")
                
                # mean dice coefficient
                avg_dice = torch.mean(dices_per_class).item()
                
                # logging with wandb
                wandb.log({
                    "Valid Mean Loss":mean_valid_loss,
                    "mDice":avg_dice
                })
                
                if best_dice < avg_dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {avg_dice:.4f}")
                    print(f"Save model in {save_dir}")
                    best_dice = avg_dice
                    save_model(model, save_dir=save_dir, file_name=args.checkpoint)
        
        if args.scheduler == "plateau":
            scheduler.step(avg_dice)
        else:
            scheduler.step()
                
             
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # hyperparameters
    parser.add_argument('--seed', type=int, default=20240216)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=2)
    parser.add_argument('--valid_interval', type=int, default=10)
    parser.add_argument('--valid_threshold', type=float, default=0.5)
   
    # directories
    parser.add_argument('--image_root', type=str, default="../data/train/DCM")
    parser.add_argument('--label_root', type=str, default="../data/train/outputs_json")
    parser.add_argument('--save_dir', type=str, default="exp")
    parser.add_argument('--wandb_name', type=str, default="default_run")
    parser.add_argument('--checkpoint', type=str, default="best.pt")
    
    # model
    parser.add_argument('--encoder_name', type=str, default='tu-xception71', help="encoder name like mobilenet_v2 or efficientnet-b7 (default : efficientnet-b7))")
    parser.add_argument('--encoder_weights', type=str, default="imagenet", help="pre-trained weights for encoder initialization (default : imagenet))")

    # dataset
    parser.add_argument('--n_splits', type=int, default="5")
    parser.add_argument('--n_fold', type=int, default="0")
    
    # augmentation
    parser.add_argument('--clahe', type=int, default=0, help='clahe augmentation')
    parser.add_argument('--cp', type=int, default=0, help='copypaste augmentation')
    
    # loss function
    parser.add_argument('--loss', type=str, default="bce")
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # scheduler
    parser.add_argument('--scheduler', type=str, default="multistep")
    
    # validation duration
    parser.add_argument('--val_every', type=int, default=1)
    args = parser.parse_args()
    
    # wandb init
    wandb.init(entity="level2_cv4_dc", project="segmentation", name=args.wandb_name, config=vars(args))
    print(args)
    main(args)
    
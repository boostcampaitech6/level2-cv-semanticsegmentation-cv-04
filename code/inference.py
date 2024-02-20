import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import os
import numpy as np 
import albumentations as A
from tqdm.auto import tqdm
import pandas as pd
import argparse
import ttach as tta

from dataset import XRayInferenceDataset
from utils import *

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def encode_mask_to_rle(mask):
    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    # RLE로 인코딩된 결과를 mask map으로 복원합니다.
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def inference(seed, model_dir, model_name, submission_name, image_root, test_batch_size, test_thr, tta_mode, clahe) :
    
    set_seed(seed)
    
    model_root = '/opt/ml/input/code/'
    model_dir = os.path.join(model_root, model_dir)
    model = torch.load(os.path.join(model_dir, model_name))
    
    # augmentation
    tf = A.Resize(1024, 1024)

    test_dataset = XRayInferenceDataset(image_root, transforms=tf, clahe=clahe)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=test_batch_size, # 2
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # defined 2 * 3  = 6 augmentations ! tta mode가 True 일 경우 적용할 transform 설정
    transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    # tta.Resize([(512,512),(640,640),(768,768),(896,896),(1024,1024)],(512,512),"nearest")
                ]
            )

    #tta mode 키기
    if tta_mode:
        model = tta.SegmentationTTAWrapper(model, transforms)

    model = model.cuda()
    model.eval()
        
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.cuda()    
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > test_thr).detach().cpu().numpy() #(outputs > test_thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
            

    # to csv
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(submission_name, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 21)')
    parser.add_argument('--model_dir', type=str, default="workspace/exp", help="load model at /opt/ml/input/code/{model_dir} (default : exp))")
    parser.add_argument('--model_name', type=str, default="best.pt", help="load model named {model_name} (default : best.pt))")

    parser.add_argument('--submission_name', type=str, default='output.csv', help='submission file name (default: output.csv)')
    parser.add_argument('--image_root', type=str, default="/opt/ml/input/data/test/DCM", help="test image root (default: /opt/ml/input/data/test/DCM)")
    
    parser.add_argument('--test_batch_size', type=int, default=2, help='input batch size for test (default: 2)')
    parser.add_argument('--test_thr', type=float, default=.5, help='test threshold (default: 0.5)')

    parser.add_argument('--tta_mode', type=bool, default=False, help='control the tta mode')
    parser.add_argument('--clahe', type=bool, default=False, help='clahe augmentation')


    args = parser.parse_args()
    inference(**args.__dict__)
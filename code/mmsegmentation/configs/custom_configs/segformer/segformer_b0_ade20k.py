# Train Segformer Mit B0
_base_ = [
    "segformer_mit-b0.py",
    "../datasets/handbone.py",
    "../handbone_runtime.py",
    '../schedules/schedule_20k_custom.py'
]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type='EncoderDecoderWithoutArgmax',
    init_cfg=dict(
        type='Pretrained',
        # load ADE20k pretrained EncoderDecoder from mmsegmentation
        checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='SegformerHeadWithoutAccuracy',
        num_classes=29,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)
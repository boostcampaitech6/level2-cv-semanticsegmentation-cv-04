_base_ = [
    './deeplabv3_unet_s5-d16.py', '../datasets/handbone.py',
    '../handbone_runtime.py', '../schedules/schedule_40k_custom.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))

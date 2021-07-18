_base_ = [
    './models/swin_transformer/small_224.py',
    './datasets/imagenet_bs64_swin_224.py',
    './schedules/imagenet_bs1024_adamw_swin.py',
    './default_runtime.py'
]
_base_ = [
    './models/resnet101.py',
    './datasets/imagenet_bs32.py',
    './schedules/imagenet_bs256.py',
    './default_runtime.py'
]

# checkpoint saving
checkpoint_config = dict(interval=10, max_keep_ckpts=5)
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = './work_dirs/resnet101_batch256_imagenet_20200708-753f3608.pth'
load_from = './work_dirs/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth'
resume_from = None
workflow = [('train', 1)]

work_dir = './work_dirs/swin-transformer'

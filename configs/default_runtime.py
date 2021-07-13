# checkpoint saving
checkpoint_config = dict(interval=10, max_keep_ckpts=5)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './work_dirs/resnet101_batch256_imagenet_20200708-753f3608.pth'
resume_from = None
workflow = [('train', 1)]

work_dir = './work_dirs/kdxf_cls_task'

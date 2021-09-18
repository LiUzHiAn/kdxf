# dataset settings
dataset_type = 'ImageWordEmbeddingDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageEmbeddingFromFile'),
    dict(type='RandomResizedCrop', size=224, backend="pillow"),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['emb']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['emb', 'img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageEmbeddingFromFile'),
    dict(type='Resize', size=(256, -1), backend="pillow"),

    # Test Time Augmentation
    # dict(type='RandomResizedCrop', size=224),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),

    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['emb']),
    dict(type='Collect', keys=['emb', 'img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='ClassBalancedDataset',
        dataset=dict(
            type=dataset_type,
            data_prefix='./data/kdxf_cls/training_set',
            ann_file='./data/kdxf_cls/train.txt',
            classes='./data/kdxf_cls/classes.txt',
            pipeline=train_pipeline),
        oversample_thr=0.02),
    # type=dataset_type,
    # data_prefix='./data/kdxf_cls/training_set',
    # ann_file='./data/kdxf_cls/train.txt',
    # classes='./data/kdxf_cls/classes.txt',
    # pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='./data/kdxf_cls/training_set',
        ann_file='./data/kdxf_cls/val.txt',
        classes='./data/kdxf_cls/classes.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='./data/kdxf_cls/test_set',
        ann_file='./data/kdxf_cls/test.txt',
        classes='./data/kdxf_cls/classes.txt',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='accuracy', metric_options=dict(topk=(1, 5)))

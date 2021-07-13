import os
import mmcv
import os.path as osp

from itertools import chain

data_root = './data/kdxf_cls'


# Generate mapping from class_name to label
def find_folders(root_dir):
    folders = [
        d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))
    ]
    folders.sort(key=int)
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


# Generate annotations
def gen_annotations(root_dir):
    annotations = dict()
    folder_to_idx = find_folders(root_dir)

    for cls_dir, label in folder_to_idx.items():
        cls_to_label = [
            '{} {}'.format(osp.join(cls_dir, filename), label)
            for filename in mmcv.scandir(osp.join(root_dir, cls_dir), suffix='.jpg')
        ]
        annotations[cls_dir] = cls_to_label
    return annotations


def train_val_split(split_frac=0.85):
    train_dir = osp.join(data_root, 'training_set')

    # Generate class list
    folder_to_idx = find_folders(train_dir)
    classes = list(folder_to_idx.keys())
    with open(osp.join(data_root, 'classes.txt'), 'w') as f:
        f.writelines('\n'.join(classes))

    # Generate train/val set randomly
    annotations = gen_annotations(train_dir)
    # Select first 85% as train set
    train_length = lambda x: int(len(x) * split_frac)
    train_annotations = map(lambda x: x[:train_length(x)], annotations.values())
    val_annotations = map(lambda x: x[train_length(x):], annotations.values())

    # Save train/val annotations
    with open(osp.join(data_root, 'train.txt'), 'w') as f:
        contents = chain(*train_annotations)
        f.writelines('\n'.join(contents))
    with open(osp.join(data_root, 'val.txt'), 'w') as f:
        contents = chain(*val_annotations)
        f.writelines('\n'.join(contents))


def gen_test_anno_file():
    # Save test annotations
    test_dir = osp.join(data_root, 'test_set')
    test_imgs = list(mmcv.scandir(test_dir, suffix='.jpg'))
    test_imgs.sort(key=lambda str: int(str[1:].split(".jpg")[0]))
    with open(osp.join(data_root, 'test.txt'), 'w') as f:
        f.writelines('\n'.join(test_imgs))


if __name__ == '__main__':
    train_val_split(split_frac=0.85)
    gen_test_anno_file()

#####################################################################################################
# Reference:                                                                                        #
# [1]: https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/imagenet_utils.py       #
# [2]: https://github.com/tensorpack/benchmarks/blob/master/ImageNet/benchmark-dataflow.py          #
# [3]: https://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html                           #
#####################################################################################################


import argparse
import os
import os.path

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from tensorpack import imgaug, dataset
from tensorpack.dataflow import TestDataSpeed as TestDataFlowSpeed
from tensorpack.dataflow.serialize import LMDBSerializer
from tensorpack.utils.utils import get_tqdm, get_tqdm_kwargs

from tools.tp_data_loader import get_sequential_loader, get_random_loader
from tools.tv_data_loader import ImageFolderLMDB


def get_tv_augmentors(isTrain):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if isTrain:
        augmentors = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        augmentors = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    return augmentors


def get_tv_loader(data_dir, name, batch_size, workers=4):
    isTrain = name == 'train'
    augmentors = get_tv_augmentors(isTrain)
    transform = transforms.Compose(augmentors)

    if data_dir.endswith('lmdb'):
        data_dir = os.path.join(data_dir, 'ILSVRC-%s.lmdb' % name)
        dset = ImageFolderLMDB(data_dir, transform)
        ds = data.DataLoader(
            dset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

    else:
        # 500000[76:25:36, 1.82it/s]
        data_dir = os.path.join(data_dir, name)
        dset = datasets.ImageFolder(data_dir, transform)
        ds = data.DataLoader(
            dset, batch_size=batch_size, shuffle=isTrain,
            num_workers=workers, pin_memory=True)

    return ds


class TestDataLoaderSpeed(TestDataFlowSpeed):
    """ Test the speed of a DataLoader """

    def start(self):
        """
        Start testing with a progress bar.
        """
        itr = self.ds.__iter__()
        if self.warmup:
            for _ in tqdm.trange(self.warmup, **get_tqdm_kwargs()):
                next(itr)
        # add smoothing for speed benchmark
        with get_tqdm(total=self.test_size,
                      leave=True, smoothing=0.2) as pbar:
            for idx, dp in enumerate(itr):
                pbar.update()
                if idx == self.test_size - 1:
                    break


def get_tp_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_CUBIC
    # linear seems to have more stable performance.
    # but we keep cubic for compatibility with old models
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            # imgaug.RandomOrderAug(
            #     [imgaug.BrightnessScale((0.6, 1.4), clip=False),
            #      imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
            #      imgaug.Saturation(0.4, rgb=False),
            #      # rgb-bgr conversion for the constants copied from fb.resnet.torch
            #      imgaug.Lighting(0.1,
            #                      eigval=np.asarray(
            #                          [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
            #                      eigvec=np.array(
            #                          [[-0.5675, 0.7192, 0.4009],
            #                           [-0.5808, -0.0045, -0.8140],
            #                           [-0.5836, -0.6948, 0.4203]],
            #                          dtype='float32')[::-1, ::-1]
            #                      )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_tp_loader(data_dir, name, batch_size, parallel=None):
    isTrain = name == 'train'
    augmentors = get_tp_augmentor(isTrain)

    if data_dir.endswith('lmdb'):
        # 500000[70:87:20, 1.95it/s]
        data_dir = os.path.join(data_dir, 'ILSVRC-%s.lmdb' % name)
        ds = LMDBSerializer.load(data_dir, shuffle=False)
        ds = get_sequential_loader(ds, isTrain, batch_size, augmentors, parallel)
    else:
        # 500000[27:11:03, 5.11it/s]
        if isTrain:
            ds = dataset.ILSVRC12(data_dir, name, shuffle=True)
        else:
            ds = dataset.ILSVRC12Files(data_dir, name, shuffle=False)
        ds = get_random_loader(ds, isTrain, batch_size, augmentors, parallel)
    return ds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='data type from tensorpack(tp) or torchvision(tv)?',
                        choices=('tv', 'tp'), default='tv')
    # data_dir = "/home/sherry/datasets/ilsvrc-lmdb/ILSVRC-train.lmdb"
    # data_dir = "/home/sherry/datasets/ilsvrc2012_t1"
    parser.add_argument('data', help='file or directory of dataset')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--name', choices=['train', 'val'], default='train')
    parser.add_argument('--procs', type=int, default=20)
    args = parser.parse_args()

    assert args.name in ['train', 'val', 'test']

    if args.type == 'tp':
        ds = get_tp_loader(args.data, args.name, args.batch, args.procs)
        TestDataFlowSpeed(ds, 500000, warmup=100).start()
    else:
        ds = get_tv_loader(args.data, args.name, args.batch, args.procs)
        TestDataLoaderSpeed(ds, 500000, warmup=100).start()

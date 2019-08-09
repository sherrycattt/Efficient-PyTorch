#####################################################################################################
# Reference:                                                                                        #
# [1]: https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/imagenet_utils.py       #
# [2]: https://github.com/tensorpack/benchmarks/blob/master/ImageNet/benchmark-dataflow.py          #
# [3]: https://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html                           #
#####################################################################################################

import multiprocessing
import os
import os.path

import cv2
import numpy as np
from tensorpack import imgaug
from tensorpack.dataflow import (RNGDataFlow, MultiProcessRunnerZMQ, BatchData,
                                 AugmentImageComponent, MultiThreadMapData,
                                 MultiProcessMapDataZMQ,
                                 MapDataComponent, LocallyShuffleData)
from tensorpack.dataflow.serialize import LMDBSerializer
from tensorpack.utils import logger, stats, serialize
from tensorpack.utils.utils import get_tqdm

# os.environ['TENSORPACK_DATASET'] = './assets/'


# copied from torchvision.datasets.folder

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(RNGDataFlow):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    """

    def __init__(self, dir, name, extensions, shuffle=None):
        assert name in ['train', 'test', 'val'], name
        dir = os.path.expanduser(dir)
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        classes, class_to_idx = find_classes(self.full_dir)
        samples = make_dataset(self.full_dir, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.full_dir + "\n"
                                                                                     "Supported extensions are: " + ",".join(
                extensions)))
        for fname, _ in samples:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        idxs = np.arange(len(self.samples))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.samples[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]


class ImageFolder(DatasetFolder):
    def __init__(self, dir, name, loader=lambda x: cv2.imread(x, cv2.IMREAD_COLOR), shuffle=None):
        super(ImageFolder, self).__init__(dir, name, extensions=IMG_EXTENSIONS, shuffle=shuffle)
        self.loader = loader
        self.imglist = self.samples

    def __iter__(self):
        for fname, label in super(ImageFolder, self).__iter__():
            im = self.loader(fname)
            assert im is not None, fname
            yield [im, label]


class BinaryFolder(DatasetFolder):
    def __iter__(self):
        for fname, label in super(BinaryFolder, self).__iter__():
            with open(fname, 'rb') as f:
                bytes = f.read()
            bytes = np.asarray(bytearray(bytes), dtype='uint8')
            yield [bytes, label]


def compute_mean_std(ds, fname):
    """
    Compute mean and std in datasets.
    Usage: compute_mean_std(ds, 'mean_std.txt')
    """
    o = stats.OnlineMoments()
    for dp in get_tqdm(ds):
        feat = dp[0]  # len x dim
        for f in feat:
            o.feed(f)
    logger.info("Writing to {} ...".format(fname))
    with open(fname, 'wb') as f:
        f.write(serialize.dumps([o.mean, o.std]))


def dump_imdb(ds, output_path, parallel=None):
    """
    Create a Single-File LMDB from raw images.
    """
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading

    def mapf(dp):
        fname, label = dp
        with open(fname, 'rb') as f:
            bytes = f.read()
        bytes = np.asarray(bytearray(bytes), dtype='uint8')
        return bytes, label

    ds = MultiThreadMapData(ds, 1, mapf, buffer_size=2000, strict=True)
    ds = MultiProcessRunnerZMQ(ds, num_proc=parallel)

    LMDBSerializer.save(ds, output_path)


def get_random_loader(
        ds, isTrain, batch_size,
        augmentors, parallel=None):
    """ DataFlow data (Random Read)
    Args:
        augmentors (list[imgaug.Augmentor]): Defaults to `fbresnet_augmentor(isTrain)`

    Returns: A DataFlow which produces BGR images and labels.

    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html
    """
    assert isinstance(augmentors, list)
    aug = imgaug.AugmentorList(augmentors)

    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading

    if isTrain:
        ds = AugmentImageComponent(ds, aug, copy=False)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = MultiProcessRunnerZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls

        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = MultiProcessRunnerZMQ(ds, 1)
    return ds


def get_sequential_loader(
        ds, isTrain, batch_size,
        augmentors, parallel=None):
    """ Load a Single-File LMDB (Sequential Read)
    Args:
        augmentors (list[imgaug.Augmentor]): Defaults to `fbresnet_augmentor(isTrain)`

    Returns: A LMDBData which produces BGR images and labels.

    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html
    """
    assert isinstance(augmentors, list)
    aug = imgaug.AugmentorList(augmentors)

    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading

    if isTrain:
        ds = LocallyShuffleData(ds, 50000)
        ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
        ds = AugmentImageComponent(ds, aug, copy=False)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = BatchData(ds, batch_size, remainder=False, use_list=True)
        ds = MultiProcessRunnerZMQ(ds, parallel)
    else:
        def mapper(data):
            im, label = data
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, label

        ds = MultiProcessMapDataZMQ(ds, parallel, mapper, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True, use_list=True)
    return ds
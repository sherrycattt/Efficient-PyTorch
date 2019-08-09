import argparse
import os.path

from tensorpack.dataflow import dataset

from tools.tp_data_loader import dump_imdb, BinaryFolder, IMG_EXTENSIONS

# import numpy as np
# from tensorpack.dataflow import MultiProcessRunnerZMQ
# from tensorpack.dataflow.serialize import LMDBSerializer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ILSVRC")
    parser.add_argument('--data_dir', type=str, default="/home/sherry/datasets/ilsvrc2012_t1")
    parser.add_argument('--split', type=str, default="val")
    parser.add_argument('--out_dir', type=str, default="/home/sherry/datasets/ilsvrc-lmdb111")
    parser.add_argument('--procs', type=int, default=20)

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.dataset == "ILSVRC":

        # class BinaryILSVRC12(dataset.ILSVRC12Files):
        #     def __iter__(self):
        #         for fname, label in super(BinaryILSVRC12, self).__iter__():
        #             with open(fname, 'rb') as f:
        #                 bytes = f.read()
        #             bytes = np.asarray(bytearray(bytes), dtype='uint8')
        #             yield [bytes, label]
        # ds = BinaryILSVRC12(args.data_dir, args.split)
        # ds = MultiProcessRunnerZMQ(ds, nr_proc=args.procs)
        # LMDBSerializer.save(ds, os.path.join(args.out_dir, '%s-%s.lmdb' % (args.dataset, args.split))

        ds = dataset.ILSVRC12Files(args.data_dir, args.split)
    else:
        ds = BinaryFolder(args.data_dir, args.split, IMG_EXTENSIONS)

    output_path = os.path.join(args.out_dir, '{}-{}.lmdb'.format(args.dataset, args.split))
    dump_imdb(ds, output_path, parallel=args.procs)

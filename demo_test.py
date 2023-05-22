import numpy as np
import argparse
from datasets.data_loader import *


def dataset_test():
    # 将90类信号 分成多个任务
    parser = argparse.ArgumentParser(description='FSSEI-DATASET-TEST')
    parser.add_argument('--datasets', type=str, default=['fs_sei'], choices=list(dataset_config.keys()), nargs='+',
                        help='Dataset or datasets used (default=%(default)s)')
    parser.add_argument('--num_tasks', type=int, default=4,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    args, extra_args = parser.parse_known_args()
    print('dataset test arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
    trn_load, val_load, tst_load, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                        args.batch_size, args.num_workers, args.pin_memory)
    print('taskcla', taskcla)
    for t, (_, ncla) in enumerate(taskcla):
        print('='*100)
        print('task: ', t)
        trn_l = trn_load[t]

        for signal, targets in trn_l:
            print('signal_shape: ', signal.shape)
            print('targets_shape: ', targets.shape)
            break


if __name__ == '__main__':
    dataset_test()


import argparse
import os
from os.path import isdir, join
import shutil
import random


def separate_val_files(train_dir, val_dir, val_fr=0.2):
    random.seed(1)
    if isdir(val_dir):
        raise ValueError(f'{val_dir} already exist.')
    os.makedirs(val_dir, exist_ok=True)

    files = os.listdir(train_dir)

    storm_ids = list(set([s.split('_', 1)[0] for s in files]))
    storm_ids = sorted(storm_ids)
    random.shuffle(storm_ids)
    print('total storms: ', len(storm_ids), 'total files: ', len(files))

    val_storms = storm_ids[:int(val_fr * len(storm_ids))]
    val_files = [file for file in files if file.split('_', 1)[0] in val_storms]
    print('validation storms: ', len(val_storms), 'val_files: ', len(val_files))

    for i, file in enumerate(val_files):
        src = join(train_dir, file)
        dst = join(val_dir, file)

        shutil.move(src, dst)
        print('\rmoved', file, 'count', i + 1, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/train')
    parser.add_argument('--val_dir', default='data/val')
    parser.add_argument('--val_fraction', default=0.2)
    args = parser.parse_args()

    separate_val_files(args.train_dir, args.val_dir, val_fr=args.val_fraction)

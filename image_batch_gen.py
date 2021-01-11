import tensorflow as tf
import os
from os.path import join
import pandas as pd
import numpy as np
import cv2
import random


class ImgBatchGen(tf.keras.utils.Sequence):
    def __init__(self, img_dir, ground_truth_file, batch_size=32):
        self.bacth_size = batch_size
        files = os.listdir(img_dir)
        label_df = pd.read_csv(ground_truth_file, index_col=0, header=0)
        self.files = [(join(img_dir, f), label_df.loc[f[:7], 'wind_speed']) for f in files if f.endswith('.jpg')]
        random.shuffle(self.files)

    def __len__(self):
        return int(np.floor(len(self.files) / self.bacth_size))

    def __getitem__(self, index):
        images, labels = [], []

        for file, label in self.files[index * self.bacth_size:(index + 1) * self.bacth_size]:
            img = cv2.imread(file, 0)
            if img is None:
                continue

            images.append(img)
            labels.append(label)

        x = np.array(images)[:, :, :, np.newaxis].astype(np.float32) / 255.0
        y = np.array(labels).astype(np.float32)

        return x, y

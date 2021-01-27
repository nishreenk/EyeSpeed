import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join


class Augment:
    def __init__(self,
                 translate_range=(0.1, 0.1),
                 rotation_range=(0, 45),
                 flip_fraction=0.5
                 ):
        self.translate_range = translate_range
        self.rotation_range = rotation_range
        self.flip_fraction = flip_fraction

    def fix_edge_patch(self, img):

        edge_pixels = np.hstack([img[:, 0].ravel(), img[:, -1].ravel(), img[0, :], img[-1, :]])
        if not (edge_pixels > 253).any():
            return img

        _, mask = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        marker = np.zeros_like(mask)
        marker[0, :] = mask[0, :]
        marker[-1, :] = mask[-1, :]
        marker[:, 0] = mask[:, 0]
        marker[:, -1] = mask[:, -1]
        mask = self.imreconstruct(marker, mask)
        augmented = cv2.subtract(img, mask)
        return augmented

    def translate(self, img):
        random_translate = (np.random.uniform(self.translate_range[0], self.translate_range[1], 2) * img.shape[:2]).astype(int)

        T = np.float32([[1, 0, random_translate[0]], [0, 1, random_translate[1]]])
        augmented = cv2.warpAffine(img, T, img.shape[:2])
        return augmented

    def rotate(self, img):
        cy, cx = img.shape[:2]
        angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
        rot_mat = cv2.getRotationMatrix2D((int(cx/2), int(cy/2)), angle, 1.0)
        augmented = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_AREA)
        return augmented

    def flip_lr(self, img):
        if np.random.uniform() < self.flip_fraction:
            return cv2.flip(img, 1)
        else:
            return img.copy()

    def flip_tb(self, img):
        if np.random.uniform() < self.flip_fraction:
            return cv2.flip(img, 0)
        else:
            return img.copy()

    @staticmethod
    def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
        kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
        while True:
            expanded = cv2.dilate(src=marker, kernel=kernel)
            cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

            # Termination criterion: Expansion didn't change the image at all
            if (marker == expanded).all():
                return expanded
            marker = expanded


def demo_augmentation():
    aug = Augment(translate_range=[-0.1, 0.1],
                  rotation_range=[-45, 45],
                  flip_fraction=0.5)

    img = cv2.imread('test_img.jpg', 0)
    h, w = img.shape[:2]
    img = img[:h, :h]

    #augmented = aug.pad(img)
    augmented = aug.fix_edge_patch(img)
    #augmented = aug.translate(img)
    #augmented = aug.rotate(img)
    #augmented = aug.flip_lr(img)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(augmented)

    plt.show()

def save_aug_images(src, dst, folds):
    files = os.listdir(src)
    files = [f for f in files if f.endswith('.jpg')]
    os.makedirs(dst, exist_ok=True)

    aug = Augment(translate_range=(-0.1, 0.1),
                  rotation_range=(-15, 15),
                  flip_fraction=0.5)

    for fold in range(folds):
        for file in files:
            img = cv2.imread(join(src, file), 0)
            if img is None:
                continue
            img = aug.fix_edge_patch(img)
            img = aug.flip_lr(img)
            img = aug.flip_tb(img)
            img = aug.translate(img)
            img = aug.rotate(img)
            cv2.imwrite(join(dst, f'{file[:-4]}.{fold}.jpg'), img)


if __name__ == '__main__':
    #demo_augmentation()

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/train')
    parser.add_argument('--dst', default='data/train_aug')
    parser.add_argument('--folds', default=10, type=int)
    args = parser.parse_args()

    save_aug_images(args.src, args.dst, args.folds)






import tensorflow as tf
import math
import numpy as np
from tensorflow.keras.utils import Sequence
import albumentations as A
import matplotlib.pyplot as plt
import cv2

"""The definition of the image augmentations"""

""" The online augmentations"""
online_Augmentations = A.Compose([
    A.RandomResizedCrop(height=32, width=32, scale=(0.15, 1.0), ratio=(3 / 4.0, 4 / 3.0), interpolation=cv2.INTER_CUBIC,
                        always_apply=True, p=1.0),
    A.HorizontalFlip(always_apply=False, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, always_apply=False, p=0.8),
    A.ToGray(always_apply=False, p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=2, always_apply=True, p=1.0),
    A.Solarize(threshold=128, always_apply=False, p=0),
    A.ToFloat(max_value=255.0, always_apply=True, p=1.0)
])

""" The target augmentations"""
target_Augmentations = A.Compose([
    A.RandomResizedCrop(height=32, width=32, scale=(0.15, 1.0), ratio=(3 / 4.0, 4 / 3.0), interpolation=cv2.INTER_CUBIC,
                        always_apply=True, p=1.0),
    A.HorizontalFlip(always_apply=False, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, always_apply=False, p=0.8),
    A.ToGray(always_apply=False, p=0.2),
    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=2, always_apply=False, p=0.1),
    A.Solarize(threshold=128, always_apply=False, p=0.2),
    A.ToFloat(max_value=255.0, always_apply=True, p=1.0)
])


class DatasetSequence(Sequence):
    """ Class which defines the generator."""

    def __init__(self, x_train, y_train, batch_size, augmentations):
        self.x, self.y = x_train, y_train
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)

    def on_epoch_end(self):
        np.random.seed(int(np.random.rand() * 1000))
        np.random.shuffle(self.x)
        np.random.shuffle(self.y)


def generator_setup(batch_size):
    """ Tis function loads two generators"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    online_gen = DatasetSequence(x_train, y_train, batch_size=batch_size, augmentations=online_Augmentations)
    target_gen = DatasetSequence(x_train, y_train, batch_size=batch_size, augmentations=target_Augmentations)
    return online_gen, target_gen


def visualize_batch(gen1, gen2):
    # Visualize random batch to confirm their validity
    x = gen1[42][0]
    y = gen2[42][0]
    print(x)

    for i in range(0, 50):
        image = x[i]
        plt.imshow(image)
        plt.show()
        image = y[i]
        plt.imshow(image)
        plt.show()

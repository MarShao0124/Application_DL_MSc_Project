# -*- coding: utf-8 -*-
# """
#   VisA_loader.py
#   2025.03.06. @MarShao0124
#   Load VisA dataset
#
#   Reference:
#   SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation,
#   Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar,
#   arXiv preprint arXiv:2207.14315
# """


############
#   IMPORT #
############
# 1. Built-in modules
import os

# 2. Third-party modules
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

# 3. Own modules


###########
#   CLASS #
###########
class VisALoader(object):
    folder_path = os.path.dirname(os.path.dirname(__file__))
    base_path = os.path.join(folder_path, 'data/VisA_20220922')

    train, test = None, None
    num_train, num_test = 0, 0

    def setup_base_path(self, path):
        self.base_path = path
    

    def load(self, category, repeat=4, max_rot=10):
        # data, mask, binary anomaly label (0 for anomaly, 1 for good)
        x, y, z = [], [], []

        # Load training set
        split_csv = pd.read_csv(os.path.join(self.base_path, 'split_csv','1cls.csv'))
        files = split_csv[(split_csv['split'] == 'train') & (split_csv['object'] == category)]['image']

        zero_mask = tf.zeros(shape=(224, 224), dtype=tf.int32)

        for rdx in range(repeat):
            for _files in files:
                full_path = os.path.join(self.base_path, _files)
                img = self._read_image(full_path=full_path)

                if not max_rot == 0:
                    img = tf.keras.preprocessing.image.random_rotation(img, max_rot)

                mask = zero_mask

                x.append(img)
                y.append(mask)
                z.append(1)

        x = np.asarray(x)
        y = np.asarray(y)
        self.num_train = len(x)

        # Convert to TensorFlow training datasets
        x = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x, dtype=tf.float32))
        y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y, dtype=tf.int32))
        z = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(z, dtype=tf.int32))
        self.train = tf.data.Dataset.zip((x, y, z))


        # anomaly data, binary anomaly label (0 for anomaly, 1 for good)
        x, y, z = [], [], []

        # Load test set
        files = split_csv[(split_csv['split'] == 'test') & (split_csv['object'] == category)][['image', 'mask']]
        for img_path, mask_path in files.values:
            full_path = os.path.join(self.base_path, img_path)
            img = self._read_image(full_path=full_path)
            if pd.isna(mask_path):
                mask = tf.zeros(shape=(224, 224), dtype=tf.int32)
                z.append(1)
            else:
                mask_path = os.path.join(self.base_path, mask_path)
                mask = cv2.resize(cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE), dsize=(256, 256))
                mask[mask != 0] = 255
                mask = mask / 255
                mask = mask[16:-16, 16:-16]
                mask = tf.convert_to_tensor(mask, dtype=tf.int32)   
                z.append(0)

            x.append(img)
            y.append(mask)
            
        x = np.asarray(x)
        y = np.asarray(y)
        self.num_test = len(x)

        x = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x, dtype=tf.float32))
        y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y, dtype=tf.int32))
        z = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(z, dtype=tf.int32))

        self.test = tf.data.Dataset.zip((x, y, z))


    @staticmethod
    def _read_image(full_path, flags=cv2.IMREAD_COLOR):
        img = cv2.imread(full_path, flags=flags)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        img = cv2.resize(img, dsize=(256, 256))

        img = img[16:-16, 16:-16, :]

        return img
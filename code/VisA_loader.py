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

    category = {}
    categories = os.listdir(base_path)
    categories.remove('.DS_Store')
    categories.remove('LICENSE-DATASET')
    categories.remove('split_csv')
    for each in categories:
        image_anno = pd.read_csv(os.path.join(base_path, each, 'image_anno.csv'))
        category[each] = image_anno[image_anno['label'] != 'normal']['image'].tolist()


    def setup_base_path(self, path):
        self.base_path = path
    

    def load(self, category, repeat=4, max_rot=10):
        # data, mask, binary anomaly label (0 for anomaly, 1 for good)
        x, y, z = [], [], []

        # Load normal set and take 90% of the data for training
        path = os.path.join(os.path.join(self.base_path, category), 'Data/Images/Normal')
        files = os.listdir(path)

        zero_mask = tf.zeros(shape=(224, 224), dtype=tf.int32)

        for rdx in range(repeat):
            for _files in files:
                full_path = os.path.join(path, _files)
                img = self._read_image(full_path=full_path)

                if not max_rot == 0:
                    img = tf.keras.preprocessing.image.random_rotation(img, max_rot)

                mask = zero_mask

                x.append(img)
                y.append(mask)
                z.append(1)

        x = np.asarray(x)
        y = np.asarray(y)

        # Slice x and y by a ratio of 0.9:0.1
        split_index = int(len(x) * 0.9)
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        z_train, z_test = z[:split_index], z[split_index:]
        self.num_train = len(x_train)

        # Convert to TensorFlow training datasets
        x_train = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_train, dtype=tf.float32))
        y_train = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_train, dtype=tf.int32))
        z_train = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(z, dtype=tf.int32))
        self.train = tf.data.Dataset.zip((x_train, y_train, z_train))


        # anomaly data, binary anomaly label (0 for anomaly, 1 for good)
        x, y, z = [], [], []

        # Load test set
        for ano_path in self.category[category]:
            full_path = os.path.join(self.base_path, ano_path)

            img = self._read_image(full_path=full_path)

            # Split the path into directory and file name
            dir_path, file_name = os.path.split(full_path)
            dir_path = dir_path.replace('Images', 'Masks')
            file_name = os.path.splitext(file_name)[0] + '.png'
            mask_path = os.path.join(dir_path, file_name)

            mask = cv2.resize(cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE), dsize=(256, 256)) / 255
            mask = mask[16:-16, 16:-16]
            mask = tf.convert_to_tensor(mask, dtype=tf.int32)

            x.append(img)
            y.append(mask)
            z.append(0)

        x = np.asarray(x)
        y = np.asarray(y)

        x = np.concatenate((x, x_test), axis=0)
        y = np.concatenate((y, y_test), axis=0)
        z += z_test

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
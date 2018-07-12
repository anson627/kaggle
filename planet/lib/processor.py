import os
import cv2

import numpy as np

from tqdm import tqdm

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input


class DataProcessor:

    root_path = ""
    input_shape = ()

    def __init__(self, root_path, input_shape):
        self.root_path = root_path
        self.input_shape = input_shape

    def process_image_input(self, csv, folder_name, labels):
        xs = []
        ys = []
        label_dict = {k: v for v, k in enumerate(labels)}
        for name, tags in tqdm(csv.values):
            img = load_img(os.path.join(self.root_path, folder_name, '{}.jpg'.format(name)),
                           target_size=self.input_shape)
            img = img_to_array(img)
            img = preprocess_input(img)
            y = np.zeros(len(labels))
            for t in tags.split(' '):
                y[label_dict[t]] = 1
            xs.append(img)
            ys.append(y)

        xs = np.array(xs)
        ys = np.array(ys)
        return xs, ys

    @staticmethod
    def process_file_input(csv, labels):
        xs = []
        ys = []
        label_dict = {k: v for v, k in enumerate(labels)}
        for name, tags in tqdm(csv.values):
            y = np.zeros(len(labels))
            for t in tags.split(' '):
                y[label_dict[t]] = 1
            xs.append(name)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def process_test_input(self, csv, folder_name):
        xs = []
        for name, _ in tqdm(csv.values):
            img = load_img(os.path.join(self.root_path, folder_name, '{}.jpg'.format(name)),
                           target_size=self.input_shape)
            img = img_to_array(img)
            img = preprocess_input(img)
            xs.append(img)
        xs = np.array(xs)
        return xs

    def process_output(self, csv, prediction, output_name, thresholds):
        res = []
        for i in tqdm(range(prediction.shape[0])):
            a = prediction.ix[[i]]
            tags = []
            for k, v in thresholds.items():
                if a[k][i] >= v:
                    tags.append(k)
            res.append(' '.join(tags))

        csv['tags'] = res
        csv.to_csv(os.path.join(self.root_path, output_name), index=False)

    def get_generator(self, zip_list, folder_name, batch_size, has_label=True, option=-1):
        batch_idx = 0
        while True:
            begin = batch_size * batch_idx
            end = batch_size * (batch_idx + 1)
            batch_input = zip_list[begin:end]
            image_list = []
            label_list = []
            batch_idx += 1
            if has_label:
                for name, label in batch_input:
                    path = os.path.join(self.root_path, folder_name, '{}.jpg'.format(name))
                    image = cv2.resize(cv2.imread(path), self.input_shape)
                    image = self.random_augment(image, option)
                    image_list.append(image)
                    label_list.append(label)
                yield (np.array(image_list), np.array(label_list))
            else:
                for name in batch_input:
                    path = os.path.join(self.root_path, folder_name, '{}.jpg'.format(name))
                    image = cv2.resize(cv2.imread(path), self.input_shape)
                    image = self.random_augment(image, option)
                    image_list.append(image)
                yield (np.array(image_list))

            if has_label and batch_idx == len(zip_list) / batch_size:
                batch_idx = 0

    @staticmethod
    def random_augment(img, opt):
        if opt == -1:
            opt = np.random.randint(6)
        rows, cols, channel = img.shape
        degree = 90
        flip = False
        if opt == 1:
            degree = 90
            flip = True
        if opt == 2:
            degree = 180
        if opt == 3:
            degree = 180
            flip = True
        if opt == 4:
            degree = 270
        if opt == 5:
            degree = 270
            flip = True

        mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
        img = cv2.warpAffine(img, mat, (cols, rows))
        if flip:
            img = cv2.flip(img, 1)
        return img

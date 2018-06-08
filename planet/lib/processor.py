import os
import cv2

import numpy as np

from tqdm import tqdm


class DataProcessor:

    root_path = ""
    input_shape = ()

    def __init__(self, root_path, input_shape):
        self.root_path = root_path
        self.input_shape = input_shape

    def process_train_input(self, csv, folder_name, labels):
        xs = []
        ys = []
        label_dict = {k: v for v, k in enumerate(labels)}
        for name, tags in tqdm(csv.values):
            img = cv2.imread(os.path.join(self.root_path, folder_name, '{}.jpg'.format(name)))
            y = np.zeros(len(labels))
            for t in tags.split(' '):
                y[label_dict[t]] = 1
                xs.append(cv2.resize(img, self.input_shape))
                ys.append(y)

        xs = np.array(xs, np.float32) / 255.
        ys = np.array(ys, np.uint8)
        return xs, ys

    def process_test_input(self, csv, folder_name):
        xs = []
        for name, _ in tqdm(csv.values):
            img = cv2.imread(os.path.join(self.root_path, folder_name, '{}.jpg'.format(name)))
            xs.append(cv2.resize(img, self.input_shape))
        xs = np.array(xs, np.uint8)
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

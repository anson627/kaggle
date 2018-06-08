import os
import cv2

import pandas as pd
import numpy as np

from tqdm import tqdm


class image_processor:

    root_path = ""
    input_shape = ()

    def __init__(self, root_path, input_shape):
        self.root_path = root_path
        self.input_shape = input_shape

    def preprocess_input(self, csv_name, folder_name, label_map):
        xs = []
        ys = []
        csv = pd.read_csv(os.path.join(self.root_path, csv_name))
        for name, tags in tqdm(csv.values, miniters=1000):
            img = cv2.imread(os.path.join(self.root_path, folder_name, '{}.jpg'.format(name)))
            labels = np.zeros(len(label_map))
            for t in tags.split(' '):
                labels[label_map[t]] = 1
                xs.append(cv2.resize(img, self.input_shape))
                ys.append(labels)

        xs = np.array(xs, np.float32) / 255.
        ys = np.array(ys, np.uint8)
        return xs, ys

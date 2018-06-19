import os
import gc

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from lib.processor import DataProcessor
from lib.classifier import ImageClassifier
# import matplotlib.pyplot as plt

root_path = "/opt/michelangelo/snapshots/"
img_size = 128
batch_size = 128
learning_rates = [0.001, 0.0001, 0.00001]
learning_epochs = [20, 5, 5]
num_splits = 5

labels = ['blow_down',
          'bare_ground',
          'conventional_mine',
          'blooming',
          'cultivation',
          'artisinal_mine',
          'haze',
          'primary',
          'slash_burn',
          'habitation',
          'clear',
          'road',
          'selective_logging',
          'partly_cloudy',
          'agriculture',
          'water',
          'cloudy']

thresholds = {k: 0.2 for v, k in enumerate(labels)}


def train():
    processor = DataProcessor(root_path, (img_size, img_size))
    classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_image_input(csv, 'train-jpg', labels)
    x_train, x_valid, y_train, y_valid = train_test_split(xs, ys, test_size=0.2, random_state=1)
    for lr, epochs in zip(learning_rates, learning_epochs):
        classifier.train(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, lr=lr,
                         epochs=epochs)


def predict():
    processor = DataProcessor(root_path, (img_size, img_size))
    classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    x_test = processor.process_test_input(csv, 'test-jpg')
    y_test = classifier.predict(x_test, batch_size=batch_size)
    y_test = pd.DataFrame(y_test, columns=labels)
    processor.process_output(csv, y_test, 'submission.csv', thresholds)


def k_fold_train():
    processor = DataProcessor(root_path, (img_size, img_size))
    classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_image_input(csv, 'train-jpg', labels)
    k_fold = KFold(n_splits=num_splits)
    idx_split = 0
    for train_index, test_index in k_fold.split(xs):
        print("split {}".format(idx_split))
        x_train, x_valid = xs[train_index], xs[test_index]
        y_train, y_valid = ys[train_index], ys[test_index]
        for lr, epochs in zip(learning_rates, learning_epochs):
            print("learning rate {}, epochs {}".format(lr, epochs))
            classifier.train(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, lr=lr,
                             epochs=epochs, idx_split=idx_split)
        idx_split += 1


def k_fold_predict():
    processor = DataProcessor(root_path, (img_size, img_size))
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    ys = []
    for idx_split in range(num_splits):
        classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))
        classifier.load(idx_split)
        x_test = processor.process_test_input(csv, 'test-jpg')
        y_test = classifier.predict(x_test, batch_size=batch_size)
        ys.append(y_test)

    prediction = np.array(ys[0])
    for idx_split in range(1, num_splits):
        prediction += np.array(ys[idx_split])
    prediction /= num_splits

    prediction = pd.DataFrame(prediction, columns=labels)
    processor.process_output(csv, prediction, 'submission.csv', thresholds)


def train_generator():
    processor = DataProcessor(root_path, (img_size, img_size))
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_file_input(csv, labels)
    k_fold = KFold(n_splits=num_splits)
    lr = 1e-4
    epochs = 50
    idx_split = 0
    for train_index, valid_index in k_fold.split(xs):
        x_train, x_valid = xs[train_index], xs[valid_index]
        y_train, y_valid = ys[train_index], ys[valid_index]
        print("split {} train size {} valid size {}".format(idx_split, len(x_train), len(x_valid)))
        gen_train = processor.get_generator(list(zip(x_train, y_train)), 'train-jpg', batch_size)
        gen_valid = processor.get_generator(list(zip(x_valid, y_valid)), 'train-jpg', batch_size)
        print("learning rate {}, epochs {}".format(lr, epochs))
        classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))
        classifier.train_generator(gen_train, len(x_train), gen_valid, len(x_valid), batch_size, lr=lr, epochs=epochs,
                                   idx_split=idx_split)
        idx_split += 1


def predict_generator():
    processor = DataProcessor(root_path, (img_size, img_size))
    classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    x_test, _ = processor.process_file_input(csv, labels)
    ps = []
    for idx_split in range(num_splits):
        classifier.load(idx_split)
        ys = []
        for opt in range(6):
            print('split {} TTA {}'.format(idx_split, opt))
            gen_test = processor.get_generator(x_test, 'test-jpg', batch_size, has_label=False, option=opt)
            y = classifier.predict_generator(gen_test, len(x_test), batch_size)
            ys.append(y)
        y = np.array(ys[0])
        for i in range(1, 6):
            y += np.array(ys[i])
        y /= 6
        ps.append(y)

    prediction = np.array(ps[0])
    for idx_split in range(1, num_splits):
        prediction += np.array(ps[idx_split])
    prediction /= num_splits

    prediction = pd.DataFrame(prediction, columns=labels)
    processor.process_output(csv, prediction, 'submission.csv', thresholds)


def main():
    train_generator()
    predict_generator()
    # train()
    # predict()


if __name__ == '__main__':
    main()

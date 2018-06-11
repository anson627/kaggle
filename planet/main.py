import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from lib.processor import DataProcessor
from lib.classifier import ImageClassifier

root_path = "/opt/michelangelo/snapshots/"
img_size = 64
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

thresholds = {'blow_down':0.2,
                'bare_ground':0.138,
                'conventional_mine':0.1,
                'blooming':0.168,
                'cultivation':0.204,
                'artisinal_mine':0.114,
                'haze':0.204,
                'primary':0.204,
                'slash_burn':0.38,
                'habitation':0.17,
                'clear':0.13,
                'road':0.156,
                'selective_logging':0.154,
                'partly_cloudy':0.112,
                'agriculture':0.164,
                'water':0.182,
                'cloudy':0.076}

processor = DataProcessor(root_path, (img_size, img_size))
classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels))


def train():
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_train_input(csv, 'train-jpg', labels)
    for lr, epochs in zip(learning_rates, learning_epochs):
        x_train, x_valid, y_train, y_valid = train_test_split(xs, ys, test_size=0.2, random_state=1)
        classifier.train(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, lr=lr,
                         epochs=epochs)


def predict():
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    x_test = processor.process_test_input(csv, 'test-jpg')
    y_test = classifier.predict(x_test, batch_size=batch_size)
    y_test = pd.DataFrame(y_test, columns=labels)
    processor.process_output(csv, y_test, 'submission.csv', thresholds)


def k_fold_train():
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_train_input(csv, 'train-jpg', labels)
    k_fold = KFold(n_splits=num_splits, shuffle=True, random_state=1)
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
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    ys = []
    for idx_split in range(num_splits):
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


def main():
    k_fold_train()
    # predict()


if __name__ == '__main__':
    main()

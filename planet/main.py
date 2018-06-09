import os

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from lib.processor import DataProcessor
from lib.classifier import ImageClassifier

root_path = "/opt/michelangelo/snapshots/"
img_size = 64
batch_size = 128
learning_rate = 0.00001
epochs = 15
splits = 10

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

processor = DataProcessor(root_path, (img_size, img_size))
classifier = ImageClassifier(root_path, (img_size, img_size, 3), len(labels), learning_rate)


def train():
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_train_input(csv, 'train-jpg', labels)
    kf = KFold(n_splits=splits, shuffle=True, random_state=1)
    split = 0
    for train_index, test_index in kf.split(xs):
        x_train, x_valid = xs[train_index], xs[test_index]
        y_train, y_valid = ys[train_index], ys[test_index]
        classifier.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs)
        classifier.save(os.path.join(root_path, 'model-{}.h5'.format(split)))
        split += 1


def predict():
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    ys = []
    for split in range(splits):
        classifier.load(os.path.join(root_path, 'model-{}.h5'.format(split)))
        x_test = processor.process_test_input(csv, 'test-jpg')
        y_test = classifier.predict(x_test, batch_size=batch_size)
        ys.append(y_test)

    preds = np.array(ys[0])
    for i in range(1, splits):
        preds += np.array(ys[i])
    preds /= splits

    preds = pd.DataFrame(preds, columns=labels)
    processor.process_output(csv, preds, 'submission.csv', thresholds)


def main():
    train()
    predict()


if __name__ == '__main__':
    main()

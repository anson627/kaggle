import os

import pandas as pd

from sklearn.model_selection import train_test_split

from lib.processor import DataProcessor
from lib.classifier import ImageClassifier

root_path = "/opt/michelangelo/snapshots/"
weight_path = os.path.join(root_path, 'model.h5')
img_size = 64
batch_size = 128
learning_rate = 0.00001
epochs = 15

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
classifier = ImageClassifier((img_size, img_size, 3), len(labels))


def train():
    csv = pd.read_csv(os.path.join(root_path, 'train_v2.csv'))
    xs, ys = processor.process_train_input(csv, 'train-jpg', labels)
    x_train, x_valid, y_train, y_valid = train_test_split(xs, ys, test_size=0.2, random_state=1)
    model = classifier.get_vgg19_model(learning_rate)
    model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs, shuffle=True)
    model.save_weights(weight_path)


def predict():
    model = classifier.get_vgg19_model(learning_rate)
    if os.path.isfile(weight_path):
        model.load_weights(weight_path)
    csv = pd.read_csv(os.path.join(root_path, 'sample_submission_v2.csv'))
    x_test = processor.process_test_input(csv, 'test-jpg')
    y_test = model.predict(x_test, batch_size=batch_size)
    y_test = pd.DataFrame(y_test, columns=labels)
    processor.process_output(csv, y_test, 'submission.csv', thresholds)


def main():
    # train()
    predict()


if __name__ == '__main__':
    main()
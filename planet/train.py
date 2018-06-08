import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from lib.processor import image_processor
from lib.classifier import image_classifier

path = "/opt/michelangelo/snapshots/"
img_size = 64
batch_size = 128
learning_rate = 0.001
epochs = 1

label_map = {'agriculture': 14,
             'artisinal_mine': 5,
             'bare_ground': 1,
             'blooming': 3,
             'blow_down': 0,
             'clear': 10,
             'cloudy': 16,
             'conventional_mine': 2,
             'cultivation': 4,
             'habitation': 9,
             'haze': 6,
             'partly_cloudy': 13,
             'primary': 7,
             'road': 11,
             'selective_logging': 12,
             'slash_burn': 8,
             'water': 15}

processor = image_processor(path, (img_size, img_size))
xs, ys = processor.preprocess_input('train_v2.csv', 'train-jpg', label_map)

x_train, x_valid, y_train, y_valid = train_test_split(xs, ys, test_size=0.2, random_state=1)

classifier = image_classifier((img_size, img_size, 3))
model = classifier.get_vgg19_model(len(label_map), learning_rate)
model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs, shuffle=True)

p_valid = model.predict(x_valid, batch_size=batch_size)
print('F-beta score is {}'.format(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')))

import os
import cv2

import numpy as np
import pandas as pd

from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

PATH = "/opt/michelangelo/snapshots/"
img_size = 64
input_shape = (img_size, img_size, 3)
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

x_train = []
y_train = []

df_train = pd.read_csv(os.path.join(PATH, 'train_v2.csv'))

for f, tags in tqdm(df_train.values, miniters=1000):
    path = os.path.join(PATH, 'train-jpg', '{}.jpg'.format(f))
    img = cv2.imread(path)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (img_size, img_size)))
    y_train.append(targets)

x_train = np.array(x_train, np.float32)/255.
y_train = np.array(y_train, np.uint8)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(VGG19(weights='imagenet', include_top=False, input_shape=input_shape))
model.add(Flatten())
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_size, epochs=epochs, shuffle=True)

p_valid = model.predict(X_valid, batch_size=batch_size)
print('F-beta score is {}'.format(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')))

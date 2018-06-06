import os
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

PATH = "data/dogbreed/"
img_size = 90
batch_size = 64

df_train = pd.read_csv(os.path.join(PATH, 'labels.csv'))
df_test = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

x_train = []
y_train = []
x_test = []
i = 0
for f, breed in tqdm(df_train.values):
    img = cv2.imread(os.path.join(PATH, 'train/{}.jpg'.format(f)))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (img_size, img_size)))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):
    img = cv2.imread(os.path.join(PATH, 'test/{}.jpg'.format(f)))
    x_test.append(cv2.resize(img, (img_size, img_size)))

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.

num_class = y_train_raw.shape[1]
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

base_model = VGG19(#weights='imagenet',
    weights = None, include_top=False, input_shape=(img_size, img_size, 3))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()
model.fit(X_train, Y_train, epochs=1, validation_data=(X_valid, Y_valid), verbose=1)
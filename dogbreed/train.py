import os
import shutil
from shutil import copyfile

import pandas as pd

from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

PATH = "../data/dogbreed/"
train_data_dir = os.path.join(PATH, 'tmp', 'train')
validation_data_dir = os.path.join(PATH, 'tmp', 'valid')

sz = 224
batch_size = 64

label_csv = pd.read_csv(os.path.join(PATH, 'labels.csv'))
test_csv = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

train_csv, valid_csv = train_test_split(label_csv, test_size=0.2, random_state=1)


def copy_data(data_dir, data_csv):
    shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    for id, label in data_csv.values:
        dir = os.path.join(data_dir, label)
        name = id + '.jpg'
        if not os.path.exists(dir):
            os.mkdir(dir)
        copyfile(os.path.join(PATH, 'train', name), os.path.join(dir, name))


copy_data(train_data_dir, train_csv)
copy_data(validation_data_dir, valid_csv)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=False,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='categorical')

base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(120, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(train_generator, train_generator.n // batch_size, epochs=3, workers=4,
        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)

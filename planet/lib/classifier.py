import os

from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import TensorBoard


class ImageClassifier:
    root_path = ""
    input_shape = ()
    output_size = 0
    model = None

    def __init__(self, root_path, input_shape, output_size, learning_rate):
        self.root_path = root_path
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        self.model.add(VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(self.output_size, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=['accuracy'])

    def fit(self, x, y, batch_size, validation_data, epochs):
        log_path = os.path.join(self.root_path, 'logs')
        tensor_board = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(x=x,
                       y=y,
                       validation_data=validation_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[tensor_board])

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def predict(self, x, batch_size):
        self.model.predict(x=x, batch_size=batch_size)

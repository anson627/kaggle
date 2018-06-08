from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential


class ImageClassifier:
    input_shape = ()
    output_size = 0

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size

    def get_vgg19_model(self, learning_rate):
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(self.output_size, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])
        return model

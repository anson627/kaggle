from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential


class image_classifier:
    input_shape = ()

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_vgg19_model(self, output_size, learning_rate):
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(output_size, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])
        return model

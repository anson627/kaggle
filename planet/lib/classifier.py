import os

from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model


class ImageClassifier:
    root_path = ""
    input_shape = ()
    output_size = 0
    model = None

    def __init__(self, root_path, input_shape, output_size):
        self.root_path = root_path
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        self.model.add(VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape))
        self.model.add(Flatten())
        self.model.add(Dense(self.output_size, activation='sigmoid'))
        plot_model(self.model, to_file=os.path.join(root_path, 'model.png'))

    def fit(self, x, y, batch_size, validation_data, lr, epochs, idx_split):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.Adam(lr=lr),
                           metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0)
        model_checkpoint = ModelCheckpoint(self.__get_weights_path(idx_split), monitor='val_loss', save_best_only=True)
        tensor_board = TensorBoard(log_dir=self.__get_tensor_board_path(), histogram_freq=0, write_graph=True,
                                   write_images=True)
        self.model.fit(x=x,
                       y=y,
                       validation_data=validation_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[early_stop, model_checkpoint, tensor_board])

    def predict(self, x, batch_size):
        return self.model.predict(x=x, batch_size=batch_size)

    def save(self, idx_split):
        self.model.save_weights(self.__get_weights_path(idx_split))

    def load(self, idx_split):
        self.model.load_weights(self.__get_weights_path(idx_split))

    def __get_tensor_board_path(self):
        return os.path.join(self.root_path, 'logs')

    def __get_weights_path(self, index):
        return os.path.join(self.root_path, 'model-{}.h5'.format(index))

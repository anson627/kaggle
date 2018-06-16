import os

from keras import backend
from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.utils import plot_model

from tensor_board import MyTensorBoard
from parallel_model import ParallelModel


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
        self.model = ParallelModel(self.model, gpus=4)
        # plot_model(self.model, to_file=os.path.join(root_path, 'model.png'))

    def train(self, x, y, batch_size, validation_data, lr, epochs, idx_split=0):
        def f2(y_true, y_pred):
            def recall(y_true, y_pred):
                true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
                possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + backend.epsilon())
                return recall

            def precision(y_true, y_pred):
                true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
                predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + backend.epsilon())
                return precision

            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 5 * ((precision * recall) / (4 * precision + recall + backend.epsilon()))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.Adam(lr=lr),
                           metrics=['accuracy', f2])

        early_stop = EarlyStopping(patience=2)
        model_checkpoint = ModelCheckpoint(self.__get_weights_path(idx_split), save_best_only=True)
        tensor_board = MyTensorBoard(log_dir=self.__get_logs_path(idx_split, lr, epochs), write_images=True)
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

    def __get_weights_path(self, idx_split):
        return os.path.join(self.root_path, 'models', 'split{}.h5'.format(idx_split))

    def __get_logs_path(self, idx_split, lr, epochs):
        return os.path.join(self.root_path, 'logs', 'split{}-lr{}-epochs{}'.format(idx_split, lr, epochs))

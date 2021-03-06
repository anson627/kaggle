import os

from keras.callbacks import TensorBoard
import tensorflow as tf


class MyTensorBoard(TensorBoard):
    val_writer = None

    def __init__(self, log_dir='./logs', **kwargs):
        training_log_dir = os.path.join(log_dir, 'train')
        super(MyTensorBoard, self).__init__(training_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'valid')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(MyTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(MyTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(MyTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

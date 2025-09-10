import os
import numpy as np
from glob import glob
from src import logger
from src.entity.config_entity import BaseModelConfig


import tensorflow as tf
from tensorflow.keras import layers as L



class VideoClassifier(tf.keras.Model):
    def __init__(self, num_classes: int):
        super().__init__()
        # --- define layers ---
        self.conv1 = L.TimeDistributed(
            L.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.pool1 = L.TimeDistributed(L.MaxPooling2D(pool_size=2))

        self.conv2 = L.TimeDistributed(
            L.Conv2D(16, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.pool2 = L.TimeDistributed(L.MaxPooling2D(pool_size=2))

        self.conv3 = L.TimeDistributed(
            L.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.pool3 = L.TimeDistributed(L.MaxPooling2D(pool_size=2))

        self.conv4 = L.TimeDistributed(
            L.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"))
        self.gap   = L.TimeDistributed(L.GlobalAveragePooling2D())

        self.lstm  = L.LSTM(64)
        self.fc    = L.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        # --- wire them together ---
        x = self.conv1(x, training=training)
        x = self.pool1(x, training=training)

        x = self.conv2(x, training=training)
        x = self.pool2(x, training=training)

        x = self.conv3(x, training=training)
        x = self.pool3(x, training=training)

        x = self.conv4(x, training=training)
        x = self.gap(x, training=training)     # [B, T, feat]

        x = self.lstm(x, training=training)    # [B, 64]
        return self.fc(x, training=training)




if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager
    logger.info("Starting data augmentation process...")
    config = ConfigurationManager().get_base_model_config()
    

    # ---- usage ----
    model = VideoClassifier(config.NUM_CLASSES)

    # build by calling once
    _ = model(tf.random.uniform([
        config.BATCH_SIZE,
        config.SEQUENCE_LENGTH,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        config.CHANNELS
    ]))

    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    model.summary()


    logger.info("Data augmentation process completed.")
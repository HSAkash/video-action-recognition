from src import logger
from src.entity.config_entity import BaseModelConfig
from src.utils.helperFunction import plot_model

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Sequential


class VideoClassifier:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            L.TimeDistributed(
                L.Conv2D(32,3,1),
                input_shape=(
                    self.config.SEQUENCE_LENGTH,
                    self.config.IMAGE_SIZE,
                    self.config.IMAGE_SIZE,
                    self.config.CHANNELS
                )
            )
        )
        self.model.add(L.TimeDistributed(L.MaxPooling2D(2)))
        self.model.add(L.TimeDistributed(L.Conv2D(16,3,1)))
        self.model.add(L.TimeDistributed(L.MaxPooling2D(2)))
        self.model.add(L.TimeDistributed(L.Conv2D(32,3,1)))
        self.model.add(L.TimeDistributed(L.MaxPooling2D(2)))
        self.model.add(L.TimeDistributed(L.Conv2D(64,3,1)))
        self.model.add(L.TimeDistributed(L.GlobalAveragePooling2D()))
        self.model.add(L.LSTM(64))
        self.model.add(L.Dense(self.config.NUM_CLASSES, activation='softmax'))

        self.model.summary()
        plot_model(self.model, self.config.model_architecture_plot_path)

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.save(self.config.base_model_path)

        logger.info(f"Model architecture saved to {self.config.model_architecture_plot_path}")




if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager
    logger.info("Building model...")
    config = ConfigurationManager().get_base_model_config()
    model_obj = VideoClassifier(config)
    model_obj.build_model()


    logger.info("Model build completed.")
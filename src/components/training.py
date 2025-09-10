import os
import pandas as pd
from glob import glob
from pathlib import Path
from src import logger
from src.entity.config_entity import TrainingConfig
from src.components.prepare_callbacks import PrepareCallbacks
from src.components.dataset_loader import DatasetLoader

from src.utils.helperFunction import plot_loss_curves_history
import tensorflow as tf

class Training:
    def __init__(self, config: TrainingConfig, callbacks, train_ds, test_ds):
        self.config = config
        self.callbacks = callbacks
        self.train_ds = train_ds
        self.test_ds = test_ds

    def load_model(self):
        save_model_paths = glob(os.path.join(self.config.checkpoint_path, '*.keras'))
        save_model_paths = sorted(save_model_paths)
        if len(save_model_paths) == 0:
            logger.info("No saved model found. Initializing a new model.")
            return tf.keras.models.load_model(self.config.base_model_path)
        return tf.keras.models.load_model(save_model_paths[-1])
    

    def _get_initial_epoch(self):
        if not os.path.exists(self.config.history_path):
            return 0
        try:
            history = pd.read_csv(self.config.history_path)
        except Exception as e:
            logger.error("Error reading history file: ", str(e))
            return 0
        return max(history['epoch'])+1 if 'epoch' in history.columns else 0

    def train_model(self):
        # Get initial epoch & save last model state
        self.initial_epoch = self._get_initial_epoch()
        self.model = self.load_model()

        logger.info("Model loaded successfully")
        logger.info(f"Traing resume from epoch: {self.initial_epoch}")


        # Train the model
        logger.info("Training started...")
        self.model.fit(
            self.train_ds,
            epochs=self.config.EPOCHS,
            initial_epoch=self.initial_epoch,
            validation_data=self.test_ds,
            callbacks=self.callbacks,
            verbose=self.config.VERBOSE
        )
        logger.info("Training completed")

        # Save the loss and accuracy curves
        logger.info("Plotting loss and accuracy curves...")
        plot_loss_curves_history(
            history_path=self.config.history_path,
            save_loss_curves=self.config.loss_curve_path,
            save_accuracy_curves=self.config.accuracy_curve_path,
            save=self.config.SAVE_PLOTS
        )


if __name__ == "__main__":
    try:
        from src.config.configuration import ConfigurationManager
        config = ConfigurationManager()
        logger.info("Start training...")
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callbacks = prepare_callbacks.get_callbacks()

        logger.info("Preparing dataset")
        prepare_dataset_config = config.get_load_dataset_config()
        prepare_dataset = DatasetLoader(config=prepare_dataset_config)
        train_ds, test_ds, _ = prepare_dataset._prepare_data()
        logger.info("Training dataset prepared")

        logger.info("Training model")
        training_config = config.get_training_config()
        training = Training(config=training_config, callbacks=callbacks, train_ds=train_ds, test_ds=test_ds)
        training.train_model()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.exception(e)
        raise e

from src.config.configuration import ConfigurationManager
from src.components.training import Training
from src.components.prepare_callbacks import PrepareCallbacks
from src.components.dataset_loader import DatasetLoader
from src import logger


STAGE_NAME = "Training"

class TrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callbacks = prepare_callbacks.get_callbacks()

        prepare_dataset_config = config.get_load_dataset_config()
        prepare_dataset = DatasetLoader(config=prepare_dataset_config)
        train_ds, test_ds, _ = prepare_dataset._prepare_data()

        training_config = config.get_training_config()
        training = Training(config=training_config, callbacks=callbacks, train_ds=train_ds, test_ds=test_ds)
        training.train_model()


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    logger.info(f">>> stage {STAGE_NAME} started")
    training_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
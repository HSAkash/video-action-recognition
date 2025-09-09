from src.config.configuration import ConfigurationManager
from src.components.data_augmentation import ImageAugmentation
from src import logger
from src.pipeline.stage_02_splitting_dataset import DatasetSplittingPipeline


STAGE_NAME = "Dataset Augmentation"

class DatasetAugmentationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_data_augmentation_config()
        dataset_augmentation = ImageAugmentation(config=config)
        dataset_augmentation.run()



if __name__ == "__main__":
    dataset_augmentation_pipeline = DatasetAugmentationPipeline()
    logger.info(f">>> stage {STAGE_NAME} started")
    dataset_augmentation_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
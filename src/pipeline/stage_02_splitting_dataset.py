from src.config.configuration import ConfigurationManager
from src.components.splitting_dataset import SplittingDataset
from src import logger


STAGE_NAME = "Dataset Splitting"

class DatasetSplittingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_splitting_dataset_config()
        dataset_splitting = SplittingDataset(config=config)
        dataset_splitting.split_dataset()



if __name__ == "__main__":
    dataset_splitting_pipeline = DatasetSplittingPipeline()
    logger.info(f">>> stage {STAGE_NAME} started")
    dataset_splitting_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
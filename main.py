from src import logger
from src.pipeline.stage_01_image_extraction import ImageExtractionPipeline
from src.pipeline.stage_02_splitting_dataset import DatasetSplittingPipeline
from src.pipeline.stage_03_data_augmentation import DatasetAugmentationPipeline

if __name__ == "__main__":
    STAGE_NAME = "Image Extraction from Videos"
    logger.info(f">>> stage {STAGE_NAME} started")
    image_extraction_pipeline = ImageExtractionPipeline()
    image_extraction_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")

    STAGE_NAME = "Dataset Splitting"
    logger.info(f">>> stage {STAGE_NAME} started")
    dataset_splitting_pipeline = DatasetSplittingPipeline()
    dataset_splitting_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")

    STAGE_NAME = "Dataset Augmentation"
    logger.info(f">>> stage {STAGE_NAME} started")
    dataset_augmentation_pipeline = DatasetAugmentationPipeline()
    dataset_augmentation_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
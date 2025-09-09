from pyprojroot import here
from src.utils.common import read_yaml
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.entity.config_entity import (
    ImageExtractionConfig,
    SplittingDatasetConfig,
    DataAugmentationConfig,
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_image_extraction_config(self) -> ImageExtractionConfig:
        config = self.config.image_extraction

        image_extraction_config = ImageExtractionConfig(
            source_dir=here(config.source_dir),
            destination_dir=here(config.destination_dir),
            image_format=config.image_format,
            IMAGE_SIZE=self.params.IMAGE_SIZE,
            SEQUENCE_LENGTH=self.params.SEQUENCE_LENGTH,
            MAX_WORKERS=self.params.MAX_WORKERS
        )

        return image_extraction_config

    def get_splitting_dataset_config(self) -> SplittingDatasetConfig:
        config = self.config.splitting_dataset

        splitting_dataset_config = SplittingDatasetConfig(
            source_dir=here(config.source_dir),
            destination_dir=here(config.destination_dir),
            split_dir_dict_path=here(config.split_dir_dict_path),
            TRAINING_RATIO=self.params.TRAINING_RATIO,
            TESTING_RATIO=self.params.TESTING_RATIO,
            SEED=self.params.SEED,
            SEQUENCE_LENGTH=self.params.SEQUENCE_LENGTH
        )

        return splitting_dataset_config
    
    def get_data_augmentation_config(self) -> DataAugmentationConfig:
        config = self.config.data_augmentation

        data_augmentation_config = DataAugmentationConfig(
            source_dir=here(config.source_dir),
            destination_dir=here(config.destination_dir),
            ROTATE_FACTORS=self.params.ROTATE_FACTORS,
            SCALE_FACTORS=self.params.SCALE_FACTORS,
            FLIP_FACTOR=self.params.FLIP_FACTOR,
            MAX_WORKERS=self.params.MAX_WORKERS,
            IMAGE_SIZE=self.params.IMAGE_SIZE
        )

        return data_augmentation_config
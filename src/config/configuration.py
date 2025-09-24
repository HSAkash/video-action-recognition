from pyprojroot import here
from src.utils.common import read_yaml
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.entity.config_entity import (
    ImageExtractionConfig,
    SplittingDatasetConfig,
    DataAugmentationConfig,
    LoadDatasetConfig,
    PrepareCallbacksConfig,
    BaseModelConfig,
    TrainingConfig,
    EvaluationConfig
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

    def get_load_dataset_config(self) -> LoadDatasetConfig:
        config = self.config.load_dataset

        load_dataset_config = LoadDatasetConfig(
            source_dir=here(config.source_dir),
            BATCH_SIZE=self.params.BATCH_SIZE,
            IMAGE_SIZE=self.params.IMAGE_SIZE,
            CHANNELS=self.params.CHANNELS,
            SEQUENCE_LENGTH=self.params.SEQUENCE_LENGTH,
            SEED=self.params.SEED,
            SHUFFLE_BUFFER_SIZE=self.params.SHUFFLE_BUFFER_SIZE
        )

        return load_dataset_config
    
    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks

        here(config.best_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        prepare_callbacks_config = PrepareCallbacksConfig(
            best_checkpoint_path = here(config.best_checkpoint_path).__str__(),
            checkpoint_path = here(config.checkpoint_path).__str__(),
            history_path = here(config.history_path).__str__(),
            VERBOSE = self.params.VERBOSE
        )

        return prepare_callbacks_config
    

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model

        here(config.base_model_path).parent.mkdir(parents=True, exist_ok=True)

        prepare_base_model_config = BaseModelConfig(
            base_model_path = here(config.base_model_path).__str__(),
            model_architecture_plot_path = here(config.model_architecture_plot_path),
            SEED = self.params.SEED,
            NUM_CLASSES = self.params.NUM_CLASSES,
            IMAGE_SIZE = self.params.IMAGE_SIZE,
            SEQUENCE_LENGTH = self.params.SEQUENCE_LENGTH,
            BATCH_SIZE = self.params.BATCH_SIZE,
            CHANNELS = self.params.CHANNELS,
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training

        training_config = TrainingConfig(
            base_model_path = here(config.base_model_path),
            history_path = here(config.history_path),
            checkpoint_path = here(config.checkpoint_path),
            loss_curve_path = here(config.loss_curve_path),
            accuracy_curve_path = here(config.accuracy_curve_path),
            SEED = self.params.SEED,
            EPOCHS = self.params.EPOCHS,
            BATCH_SIZE = self.params.BATCH_SIZE,
            VERBOSE = self.params.VERBOSE,
            SAVE_PLOTS = self.params.SAVE_PLOTS
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation

        evaluation_config = EvaluationConfig(
            best_model_path = here(config.best_model_path),
            confusion_matrix_path = here(config.confusion_matrix_path),
            classification_report_path = here(config.classification_report_path),
            VERBOSE = self.params.VERBOSE
        )

        return evaluation_config
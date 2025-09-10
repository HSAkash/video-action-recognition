from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageExtractionConfig:
    source_dir              : Path
    destination_dir         : Path
    image_format            : str
    IMAGE_SIZE              : int
    SEQUENCE_LENGTH         : int
    MAX_WORKERS             : int


@dataclass(frozen=True)
class SplittingDatasetConfig:
    source_dir              : Path
    destination_dir         : Path
    split_dir_dict_path     : Path
    TRAINING_RATIO          : float
    TESTING_RATIO           : float
    SEED                    : int
    SEQUENCE_LENGTH         : int


@dataclass(frozen=True)
class DataAugmentationConfig:
    source_dir              : Path
    destination_dir         : Path
    ROTATE_FACTORS          : list[int] # [0, 90, 180, 270]
    SCALE_FACTORS           : list[float] # [0.5, 1.0, 1.5]
    FLIP_FACTOR             : bool
    MAX_WORKERS             : int
    IMAGE_SIZE              : int


@dataclass(frozen=True)
class LoadDatasetConfig:
    source_dir              : Path
    BATCH_SIZE              : int
    IMAGE_SIZE              : int
    CHANNELS                : int
    SEQUENCE_LENGTH         : int
    SEED                    : int
    SHUFFLE_BUFFER_SIZE     : int


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    best_checkpoint_path    : Path
    checkpoint_path         : Path
    history_path            : Path
    VERBOSE                 : bool


@dataclass(frozen=True)
class BaseModelConfig:
    base_model_path                     : Path
    model_architecture_plot_path        : Path
    SEED                                : int
    NUM_CLASSES                         : int
    IMAGE_SIZE                          : int
    SEQUENCE_LENGTH                     : int
    BATCH_SIZE                          : int
    CHANNELS                            : int


@dataclass(frozen=True)
class TrainingConfig:
    base_model_path         : Path
    history_path            : Path
    checkpoint_path         : Path
    loss_curve_path         : Path
    accuracy_curve_path     : Path
    EPOCHS                  : int
    BATCH_SIZE              : int
    SEED                    : int
    VERBOSE                 : int
    SAVE_PLOTS              : bool


@dataclass(frozen=True)
class EvaluationConfig:
    best_model_path                 : Path
    confusion_matrix_path           : Path
    classification_report_path      : Path
    VERBOSE                         : bool
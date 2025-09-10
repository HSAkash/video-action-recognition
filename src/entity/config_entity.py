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
    VERBOSE                 : int
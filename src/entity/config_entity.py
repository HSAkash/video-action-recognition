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
    TRAINING_RATIO          : float
    TESTING_RATIO           : float
    SEED                    : int
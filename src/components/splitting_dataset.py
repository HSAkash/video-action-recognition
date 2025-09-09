import shutil
from tqdm import tqdm
from src import logger
from sklearn.model_selection import train_test_split
from src.entity.config_entity import SplittingDatasetConfig

class SplittingDataset:
    def __init__(self, config: SplittingDatasetConfig):
        self.config = config
        self.train_dir = self.config.destination_dir / "train"
        self.test_dir = self.config.destination_dir / "test"

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    
    def split_dataset(self):
        class_dirs = sorted(self.config.source_dir.glob("*"))

        for class_dir in tqdm(class_dirs, desc="Splitting dataset"):
            class_name = class_dir.name
            image_sequences = sorted(class_dir.glob("*"))

            train_seqs_dir, test_seqs_dir = train_test_split(
                image_sequences,
                test_size=self.config.TESTING_RATIO,
                random_state=self.config.SEED,
                shuffle=True,
            )

            # Function to copy sequences to the respective directory
            def copy_sequences(sequences_dir, dest_dir):
                for seq_dir in sequences_dir:
                    dest_class_dir = dest_dir / class_name
                    dest_class_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(seq_dir, dest_class_dir / seq_dir.name)

            copy_sequences(train_seqs_dir, self.train_dir)
            copy_sequences(test_seqs_dir, self.test_dir)

if __name__ == "__main__":
    try:
        from src.config.configuration import ConfigurationManager
        logger.info("Starting dataset splitting process...")
        config = ConfigurationManager().get_splitting_dataset_config()
        # delete existing split dataset directory if it exists
        if config.destination_dir.exists():
            shutil.rmtree(config.destination_dir)
        dataset_splitter = SplittingDataset(config)
        dataset_splitter.split_dataset()
        logger.info("Dataset splitting completed successfully.")
    except Exception as e:
        logger.exception(e)
        raise e

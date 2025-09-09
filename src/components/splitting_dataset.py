import shutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src import logger
from sklearn.model_selection import train_test_split
from src.entity.config_entity import SplittingDatasetConfig

class SplittingDataset:
    def __init__(self, config: SplittingDatasetConfig):
        self.config = config
        shutil.rmtree(self.config.destination_dir, ignore_errors=True)
        self.train_dir = self.config.destination_dir / "train"
        self.test_dir = self.config.destination_dir / "test"

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.split_dir_dict = defaultdict(list)

        if self.config.split_dir_dict_path.exists():
            self.split_dir_dict = np.load(self.config.split_dir_dict_path, allow_pickle=True).item()

    
    def get_train_test_seqs_dir(self, image_sequences, class_name):
        train_seqs_dir = self.split_dir_dict.get(class_name, {}).get("train", [])
        test_seqs_dir = self.split_dir_dict.get(class_name, {}).get("test", [])
        if len(train_seqs_dir) == 0:
            train_seqs_dir, test_seqs_dir = train_test_split(
                image_sequences,
                test_size=self.config.TESTING_RATIO,
                random_state=self.config.SEED,
                shuffle=True,
            )
            self.split_dir_dict[class_name] = {
                "train": train_seqs_dir,
                "test": test_seqs_dir
            }
        return train_seqs_dir, test_seqs_dir

    def split_dataset(self):
        class_dirs = sorted(self.config.source_dir.glob("*"))

        for class_dir in tqdm(class_dirs, desc="Splitting dataset"):
            class_name = class_dir.name
            image_sequences = sorted(class_dir.glob("*"))

            train_seqs_dir, test_seqs_dir = self.get_train_test_seqs_dir(image_sequences, class_name)

            # Function to copy sequences to the respective directory
            def copy_sequences(sequences_dir, dest_dir):
                for seq_dir in sequences_dir:
                    dest_class_dir = dest_dir / class_name
                    try:
                        if int(sorted((dest_class_dir / seq_dir.name).glob("*"))[-1].stem) == self.config.SEQUENCE_LENGTH - 1:
                            continue
                    except IndexError:
                        pass
                    dest_class_dir.mkdir(parents=True, exist_ok=True)

                    shutil.copytree(seq_dir, dest_class_dir / seq_dir.name)

            copy_sequences(train_seqs_dir, self.train_dir)
            copy_sequences(test_seqs_dir, self.test_dir)
        np.save(self.config.split_dir_dict_path, self.split_dir_dict)

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

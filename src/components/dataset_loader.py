from src import logger
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.entity.config_entity import LoadDatasetConfig
from glob import glob


class DatasetLoader:
    def __init__(self, config: LoadDatasetConfig):
        self.config = config
        self.train_dir = self.config.source_dir / "train"
        self.test_dir = self.config.source_dir / "test"
        self._get_class_name()
        np.random.seed(self.config.SEED)
        tf.random.set_seed(self.config.SEED)


    def _get_class_name(self):
        """
        Get the class names and the number of classes in the dataset.
        """
        self.class_names = [ x.stem for x in  sorted(self.train_dir.glob("*"))]
        self.class_dict = {x:i for i,x in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def _get_labels(self, dirs: list[Path]) -> np.array:
        """
        get the labels of the dataset
        args:
            dirs: list of directories containing the dataset files
        returns:
            labels: numpy array containing the labels of the dataset
        """
        labels = np.array([self.class_dict[d.parent.name] for d in dirs])
        return labels
    

    def _prepare_data(self):
        def load_npy_files(dir_path):
            dir_path = dir_path.decode("utf-8")
            images_path = sorted(glob(f"{dir_path}/*"))
            images = []
            for image_path in images_path:
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=self.config.CHANNELS)
                image = tf.image.resize(image, [self.config.IMAGE_SIZE, self.config.IMAGE_SIZE])
                image = tf.cast(image, tf.float32) / 255.0
                images.append(image)
            images = tf.convert_to_tensor(images, dtype=tf.float32)
            return images




        def create_dataset(video_dirs, labels, batch_size):
            def load_data(video_dir, label):
                # Load mesh and keypoints data from files
                images_data = tf.numpy_function(load_npy_files, [video_dir], tf.float32)    
                # Set the shape explicitly after loading the data
                images_data.set_shape((
                    self.config.SEQUENCE_LENGTH,
                    self.config.IMAGE_SIZE,
                    self.config.IMAGE_SIZE,
                    self.config.CHANNELS
                ))  # Assuming your mesh data has this shape
                return images_data, label

            # Create a dataset from file paths and labels
            dataset = tf.data.Dataset.from_tensor_slices((video_dirs, labels))
            dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset


        X_train_dir_paths = list(self.train_dir.glob("*/*"))
        # shuffle
        np.random.shuffle(X_train_dir_paths)
        train_labels = self._get_labels(X_train_dir_paths)
        X_train_dir_paths = np.array([x.__str__() for x in X_train_dir_paths])

        X_test_dir_paths = list(self.test_dir.glob("*/*"))
        test_labels = self._get_labels(X_test_dir_paths)
        X_test_dir_paths = np.array([x.__str__() for x in X_test_dir_paths])

        # one hot encode the labels
        y_train = tf.one_hot(train_labels, depth=self.num_classes)
        y_test = tf.one_hot(test_labels, depth=self.num_classes)

        train_ds = create_dataset(X_train_dir_paths, y_train, self.config.BATCH_SIZE)

        test_ds = create_dataset(X_test_dir_paths, y_test, self.config.BATCH_SIZE)
        return train_ds, test_ds
    

if __name__ == "__main__":
    try:
        from src.config.configuration import ConfigurationManager
        logger.info("Load dataset")
        config = ConfigurationManager().get_load_dataset_config()
        image_extractor = DatasetLoader(config)
        train_ds, test_ds = image_extractor._prepare_data()
        for x, y in test_ds.take(1):
            logger.info(f"X data shape: {x.shape}")
            logger.info(f"y shape: { y.shape}")

        logger.info("Image extraction completed successfully.")
    except Exception as e:
        logger.exception(e)
        raise e
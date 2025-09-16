import cv2
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.entity.config_entity import DataAugmentationConfig

class ImageAugmentation:
    def __init__(self, config: DataAugmentationConfig, resume: bool = True):
        self.config = config
        self.resume = resume
        if not self.resume:
            if self.config.destination_dir.exists():
                shutil.rmtree(self.config.destination_dir)

    def rotate_image(self, image, angle):
        """
        Rotate an image by the specified angle.

        Args:
            image: The input image as a NumPy array.
            angle: The angle to rotate the image.

        Returns:
            The rotated image.
        """
        # Get the image dimensions
        height, width = image.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Perform the rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        return rotated_image

    def flip_image(self, image):
        """
        Flip an image horizontally.

        Args:
            image: The input image as a NumPy array.

        Returns:
            The flipped image.
        """
        return cv2.flip(image, 1)
    

    def process_image(self, img):
        h, w = img.shape[:2]
        pad_h = max(0, (self.config.IMAGE_SIZE - h) // 2)
        pad_w = max(0, (self.config.IMAGE_SIZE - w) // 2)
        img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
        crop_h = max(0, (h-self.config.IMAGE_SIZE) // 2)
        crop_w = max(0, (w-self.config.IMAGE_SIZE) // 2)
        img = img[crop_h:self.config.IMAGE_SIZE+crop_h, crop_w:self.config.IMAGE_SIZE+crop_w, :]
        return img

    def scale_image(self, image, scale_factor):
        """
        Scale an image by the specified factor.

        Args:
            image: The input image as a NumPy array.
            scale_factor: The factor by which to scale the image.

        Returns:
            The scaled image.
        """
        # Get the image dimensions
        height, width = image.shape[:2]

        # Calculate the new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Resize the image
        scaled_image = cv2.resize(image, (new_width, new_height))

        return self.process_image(scaled_image)
    

    def get_dest_path(self, img_path: Path, source_dir: Path, dist_dir: Path, aug_type: str, extra_arg: str = '') -> Path:
        """
        Return the destination path of the augmented image.

        Args:
            img_path: Path of the image.
            source_dir: The source directory of the image.
            dist_dir: The destination directory of the augmented image.
            aug_type: The type of augmentation.
            extra_arg: Extra argument for the augmented image.
        """
        # Get relative path of image w.r.t source directory
        relative_path = img_path.relative_to(source_dir)

        # Modify the parent folder name
        parent = relative_path.parent
        new_parent = parent.with_name(f"{parent.name}_{aug_type}_{extra_arg}")

        # Combine with destination directory
        return dist_dir / new_parent / relative_path.name


    def apply_augmentation(self, img_path: Path):
        '''
        This function will apply the augmentation to the images.
        Args:
            img_path: The path of the image.
            aug_dir: The path of the augmented image directory.
        '''

        # check if the image is already augmented
        dest_path = self.get_dest_path(
                img_path=img_path,
                source_dir=self.config.source_dir,
                dist_dir=self.config.destination_dir,
                aug_type='flip')
        
        if dest_path.exists():
            return


        img = cv2.imread(img_path)
        # Rotate the image by the specified angles
        for angle in self.config.ROTATE_FACTORS:
            dest_path = self.get_dest_path(
                img_path=img_path,
                source_dir=self.config.source_dir,
                dist_dir=self.config.destination_dir,
                aug_type='rotate',
                extra_arg=f"{angle}")
            rotated_image = self.rotate_image(img, angle)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(dest_path, rotated_image)

        # Scale the image by the specified factors
        for scale_factor in self.config.SCALE_FACTORS:
            dest_path = self.get_dest_path(
                img_path=img_path,  # Change to keyword argument
                source_dir=self.config.source_dir,
                dist_dir=self.config.destination_dir,
                aug_type='scale',
                extra_arg=scale_factor)
            scaled_image = self.scale_image(img, scale_factor)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(dest_path, scaled_image)
        
        # Flip the image horizontally
        if self.config.FLIP_FACTOR:
            dest_path = self.get_dest_path(
                img_path=img_path,
                source_dir=self.config.source_dir,
                dist_dir=self.config.destination_dir,
                aug_type='flip')
            flipped_image = self.flip_image(img)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(dest_path, flipped_image)
    
    def run(self):
        '''
        This function will apply the augmentation to the images.
        '''
        # Get the list of image files.
        image_files = list(self.config.source_dir.joinpath('train').glob('*/*/*'))
        # Create a list to store the tasks.
        tasks = []

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # Iterate over each image file.
            for image_file in image_files:
                tasks.append(
                    executor.submit(self.apply_augmentation, image_file)
                )

            # Wait for the tasks to complete.
            for task in tqdm(as_completed(tasks), total=len(tasks), desc="Applying Augmentation"):
                task.result()

        # Copying main images
        shutil.copytree(self.config.source_dir, self.config.destination_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    from src.config.configuration import ConfigurationManager
    logger.info("Starting data augmentation process...")
    config = ConfigurationManager().get_data_augmentation_config()
    augmentor = ImageAugmentation(config)
    augmentor.run()
    logger.info("Data augmentation process completed.")
from src.config.configuration import ConfigurationManager
from src.components.image_extraction import ImageExtraction
from src import logger


STAGE_NAME = "Image Extraction from Videos"

class ImageExtractionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_image_extraction_config()
        image_extraction = ImageExtraction(config=config)
        image_extraction.run()



if __name__ == "__main__":
    image_extraction_pipeline = ImageExtractionPipeline()
    logger.info(f">>> stage {STAGE_NAME} started")
    image_extraction_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
from src.config.configuration import ConfigurationManager
from src.components.base_model import VideoClassifier
from src import logger


STAGE_NAME = "Build model"

class BuildModelPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager().get_base_model_config()
        model_obj = VideoClassifier(config)
        model_obj.build_model()




if __name__ == "__main__":
    image_extraction_pipeline = BuildModelPipeline()
    logger.info(f">>> stage {STAGE_NAME} started")
    image_extraction_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
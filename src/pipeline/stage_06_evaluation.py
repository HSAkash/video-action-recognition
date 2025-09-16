from src.config.configuration import ConfigurationManager
from src.components.dataset_loader import DatasetLoader
from src.components.evaluation import Evaluation
from src import logger


STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()

        prepare_dataset_config = config.get_load_dataset_config()
        prepare_dataset = DatasetLoader(config=prepare_dataset_config)
        train_ds, test_ds, class_names = prepare_dataset._prepare_data()

        test_config = config.get_evaluation_config()
        evaluation = Evaluation(test_config, test_ds, class_names)
        evaluation.run()




if __name__ == "__main__":
    image_extraction_pipeline = EvaluationPipeline()
    logger.info(f">>> stage {STAGE_NAME} started")
    image_extraction_pipeline.run()
    logger.info(f">>> stage {STAGE_NAME} completed")
## train pipeline should run the data pipeline first and then should train the model

# 1st step will be running the data pipeline

import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.model_trainer = ModelTrainer()      
        
    def run_train_pipeline(self, train_arr, test_arr):
        try:
            logging.info("starting model training pipeline")   
            best_model = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model training completed. Best model on test dataset: {best_model}")
            return best_model
        except Exception as e:
            raise CustomException(e, sys)
        
        
        



from cgi import test
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class DataPipeline():
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
    

    
    def run_data_pipeline(self):
        try:
            logging.info("Starting data Pipeline...")
            train_path, test_path =self.data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed. Starting data transformation...")
            train_arr_, test_arr_, _ = self.data_transformation.initiate_data_transformation(train_path, test_path)
            return train_arr_, test_arr_
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
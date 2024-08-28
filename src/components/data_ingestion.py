import os
import sys
from unittest.mock import DEFAULT
from src.exception import CustomException
from typing import Any
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
import configparser

config = configparser.ConfigParser()
config.read('src\config.ini')
DEFAULT_PATH = config["DEFAULT_PATH"]["folder"]

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(DEFAULT_PATH,'train.csv')
    test_data_path: str = os.path.join(DEFAULT_PATH, "test.csv")
    raw_data_path: str = os.path.join(DEFAULT_PATH, "data.csv")
    
class DataIngestion():
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        
    def initiate_data_ingestion(self,):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv') ## reading
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok =True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True) ## writing
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)  ## writing
            logging.info("data ingestion is done")
            return(
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
    
    # def load(self, file_path: str):
    #     "Load data from the source folder"
    #     logging.info("Loading the data from the source folder")
    #     if not os.path.exists(file_path):
    #         raise  FileNotFoundError("The file doesnt exists")
        
    #     source_data = pd.read_csv(file_path)
    #     logging.info("Load Completed")
    #     return source_data
    
    # def serialize(self, data: Any, file_path: str):
    #     "Serialize the data"
        
        
if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr_, test_arr_, _ = data_transformation.initiate_data_transformation(
                                        data_ingestion_obj.ingestion_config.train_data_path, 
                                        data_ingestion_obj.ingestion_config.test_data_path
                                        )
    model_trainer = ModelTrainer()
    
    best_model =model_trainer.initiate_model_trainer(train_arr_,test_arr_)
    print(f"The best model:{best_model}")
    
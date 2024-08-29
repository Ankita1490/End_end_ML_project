### train different problems
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from typing import Dict
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_model, train_test_split, creating_model_parameters
import configparser
from collections import OrderedDict

config = configparser.ConfigParser(dict_type=OrderedDict)
config.read('src\config.ini')
DEFAULT_PATH = config["DEFAULT_PATH"]["folder"]


@dataclass
class Model_trainingConfig:
    trained_model_file_path = os.path.join(DEFAULT_PATH, "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = Model_trainingConfig()
        
    def initiate_model_trainer(self, train_array, test_array)->Dict :
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = train_test_split(train_array, test_array)
            logging.info(f"X_train:{X_train.shape},y_train:{y_train.shape}, X_test:{X_test.shape}, y_test:{y_test.shape}")
            models ={
                "randomforest": RandomForestRegressor(),
                "decisiontree": DecisionTreeRegressor(),
                "gradientboosting": GradientBoostingRegressor(),
                "linearregression" : LinearRegression(),
                "xgbregressor":XGBRegressor(),
                "adaboostregressor": AdaBoostRegressor(),
            }
            model_parameter = config["MODEL_PARAMETER"]
            params = creating_model_parameters(model_parameter)
            model_report:dict = evalute_model(X_train=X_train, y_train=y_train, X_test= X_test, y_test = y_test, models = models,param = params)
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = [key for key, values in model_report.items() if  values == best_model_score][0]
            best_model = {best_model_name :model_report[best_model_name]}
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model on test dataset {best_model}")
            print(models[best_model_name])
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]
            )
            return best_model
        except Exception as e:
            raise CustomException(e,sys)
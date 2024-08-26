### train different problems
import os
from statistics import mode
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_model


@dataclass
class Model_trainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = Model_trainingConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split trainng and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models ={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            
            model_report:dict = evalute_model(X_train=X_train, y_train=y_train, X_test= X_test, y_test = y_test, models = models)
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = [key for key, values in model_report.items() if  values == best_model_score][0]
            best_model = {best_model_name :model_report[best_model_name]}
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model on test dataset {best_model}")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return best_model
        except Exception as e:
            raise CustomException(e,sys)
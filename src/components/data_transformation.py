#read the features
#transformation for cat_features and numerical_features
from cgi import test
from dataclasses import dataclass
import os
import re
import sys
import pandas as pd
import numpy as np
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
     
    def get_data_transformer_obj(self, categorical_feature:List, numerical_feature:List):
        """,
        This function is responsible for data transformation
        """
        try:

            num_pipeline = Pipeline(
                steps =[
                ("imputer",SimpleImputer(strategy = "median")),
                ("scaler", StandardScaler())
                ]
            )
            logging.info("numerical columns transformation completed")
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                transformers =[
                    ("num_pipeline", num_pipeline, numerical_feature),
                    ("cat_pipeline", cat_pipeline, categorical_feature)
                ]
            )
        except Exception as e:
            raise CustomException(e, sys)
        return preprocessor
        
    def read_the_data(self, train_path, test_path):
        logging.info("Reading data from artifacts")
        try:
            self.train_set = pd.read_csv(train_path)
            self.test_set = pd.read_csv(test_path)
            logging.info("Reading the data from artifacts is completed")
        except Exception as e:
            raise CustomException(e, sys)
        

        
    def type_based_features(self):  
        try: 
            logging.info("Type based feature started")
            cat_features = [name for name in self.train_set.columns  if self.train_set[name].dtype == 'O']
            numerical_features =[name for name in self.train_set.columns  if self.train_set[name].dtype != 'O']
            taget_column_name = "math score"
            numerical_features.remove(taget_column_name)
            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"numerical columns: {numerical_features}")
            logging.info(f"Type based feature completed")
        except Exception as e:
            raise CustomException(e,sys)
        
        return cat_features, numerical_features, taget_column_name

    def initiate_data_transformation(self, train_path, test_path):
        try:
            self.read_the_data(train_path, test_path)
            logging.info("Obtaining preprocessor object")
            cat_features,numerical_features, target_column_name =self.type_based_features()
            input_feature_train_df = self.train_set.drop(columns = [target_column_name],axis =1)
            target_feature_train_df = self.train_set[target_column_name]
            
            input_feature_test_df = self.test_set.drop(columns = [target_column_name],axis =1)
            target_feature_test_df = self.test_set[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            prepocessor_obj = self.get_data_transformer_obj(categorical_feature=cat_features, numerical_feature=numerical_features)
            input_feature_train_arr = prepocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = prepocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
                        
            save_object(
                file_path =self.data_transformation_config.preprocessor_obj_file_path,
                obj = prepocessor_obj
                
            )
            logging.info("saved preprocessing obj")
            
            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
        
        
        
    

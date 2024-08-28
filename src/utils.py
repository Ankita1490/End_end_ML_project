""""comman functionality"""

import sys
import os
import  numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
 
def train_test_split(train_arr, test_arr):
    X_train, y_train, X_test, y_test = (
        train_arr[:,:-1],
        train_arr[:,-1],
        test_arr[:,:-1],
        test_arr[:,-1]
    )
    return X_train, y_train, X_test, y_test

def creating_model_parameters(model_parameter: configparser):

    params ={}
    for key, values in model_parameter.items():
        model_name, param = key.split('_', 1)
        values = values.split(',')
        if model_name not in params:
            params[model_name] ={}
            
        params[model_name][param] = [float(val) if '.' in val else int(val) if val.isdigit() else val for val in values]
    return params

def evalute_model(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}
             
        for key in models.keys():
            model = models[key]
            para = param[key]
            
            gs = GridSearchCV(model, para, cv = 3)
            gs.fit(X_train,y_train) # Train model
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            _ = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)        
            report[key] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
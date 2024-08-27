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

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
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
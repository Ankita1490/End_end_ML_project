import sys
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException



if __name__ == "__main__":
    try: 
        data_pipeline = DataPipeline()
        
        train_arr_, test_arr_= data_pipeline.run_data_pipeline()
        train_pipeline = TrainPipeline()
        best_model = train_pipeline.run_train_pipeline(train_arr_, test_arr_)
        print(f"Best model on test dataset: {best_model}")
    except Exception as e:
        raise CustomException(e, sys)
    
    
    
    
import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
#from src.components.data_ingestion import DataIngestion
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.utils import save_object


""" obj=DataIngestion()
train_data_path,test_data_path = obj.initiate_data_ingestion()
logging.info("Train and test Paths recording")

train_data = pd.read_csv(train_data_path)
logging.info("Train data reading completed")
test_data = pd.read_csv(test_data_path)
logging.info("Test data reading completed")

print(train_data.shape,test_data.shape)
logging.info("Printing train and test shapes") """

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        ''' This function is responsible for data transformation'''
        try:
            numerical_features=['writing_score','reading_score']
            categorical_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
        #Handling Missing value by median
        #Normalizing data using standard scaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical Columns standard scaling completed")
#Pipeline transformation
#Apply SimpleImputer and OneHotEncoder and StandarScaler
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical Columns encoding completed")

            #Creating entier pipeline numerical and categorical

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            ) 
            logging.info("Entire Pipeline completed")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_features=['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            output_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            output_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocesing object on training datadrame and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(output_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(output_feature_test_df)
            ]

            logging.info("Saved preprocessing object .")

            save_object(file_path=self.data_transformation_config.preprocessor_ob_file_path,
                        obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
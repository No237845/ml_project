import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

""" def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            #Train model
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
     """

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            # Hyperparameters du modèle
            param_grid = params[model_name]

            if param_grid:
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring="r2",
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train)

                # Meilleur modèle
                model = grid_search.best_estimator_

                # Affichage des meilleurs paramètres
                logging.info(
                    f"{model_name} best parameters: "
                    f"{grid_search.best_params_}"
                )

            else:

                model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
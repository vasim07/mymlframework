from .celery import app
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib

import mlflow
from mlflow import log_params, log_metrics, log_artifact, experiments

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

def log_regression_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    rsq = r2_score(actual, pred)
    return {"rmse" : rmse, "mse" : mse, "Rsq" : rsq}

xgparams = {"learning_rate" : [0.01, 0.1, 0.3], "n_estimators" : [100, 500, 1000]}
    
@app.task
def train_xgboost(X_data, y_data, params = xgparams):
    print("Starting xgboost")
    unpack_params = ParameterGrid(params)
    #mlflow.xgboost.autolog()

    for i in unpack_params:
        with mlflow.start_run():
            X_train = pd.read_json(X_data)
            y_train = pd.read_json(y_data, typ="series")
            xgb_model = xgb.XGBRegressor(**i)
            xgb_model.fit(X_train, y_train)
            pred = xgb_model.predict(X_train)
            log_metrics(log_regression_metrics(y_train, pred))    
            log_params(i)
            
rf_params = {"n_estimators":[100, 1000, 2000], "max_depth":[2,5,10]}    

@app.task
def train_rf(X_data, y_data, params=rf_params):
    print("Starting rf")
    unpack_params = ParameterGrid(params)

    for i in unpack_params:
        with mlflow.start_run():
            X_train = pd.read_json(X_data)
            y_train = pd.read_json(y_data, typ="series")
            rf_model = RandomForestRegressor(**i)
            rf_model.fit(X_train, y_train)
            pred = rf_model.predict(X_train)
            log_metrics(log_regression_metrics(y_train, pred))
            log_params(i)
        
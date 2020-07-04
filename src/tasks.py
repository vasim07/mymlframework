from .celery import app
import pandas as pd
import xgboost as xgb
import os
import joblib

import mlflow
from mlflow import log_params, log_metrics, log_artifact, experiments

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('from tasks.py!')

@app.task
def train_xgboost(X_train, y_train):
    with mlflow.start_run():
        X_train = pd.read_json(X_train)
        y_train = pd.read_json(y_train, typ="series")
        xgb_model = xgb.XGBRegressor(learning_rate=0.01, n_estimators = 1000)
        xgb_model.fit(X_train, y_train)
        pred = xgb_model.predict(X_train)

        mse = mean_squared_error(y_train, pred)
        rsq = r2_score(y_train, pred)
        log_metrics({'mean_squared_error' : mse, "RSquare" : rsq})
        #log_artifact(xgb_model)
        #print(xgb_model)
        print("xgboost compelted!")

@app.task
def train_rf(X_train, y_train):
    with mlflow.start_run():
        X_train = pd.read_json(X_train)
        y_train = pd.read_json(y_train, typ="series")
        rf_model = RandomForestRegressor(n_estimators=1000)
        rf_model.fit(X_train, y_train)
        pred = rf_model.predict(X_train)

        mse = mean_squared_error(y_train, pred)
        rsq = r2_score(y_train, pred)
        log_metrics({'mean_squared_error' : mse, "RSquare" : rsq})
        #log_artifact(rf_model)
        print("rf compelted!")
    
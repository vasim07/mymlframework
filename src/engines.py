from src.celery import app
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib

import mlflow
from mlflow import log_params, log_metrics, log_artifact, experiments

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, plot_roc_curve, accuracy_score

from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, LogisticRegression


# xgparams = os.environ.get("xgparams")
# rf_params = os.environ.get("rf_params")
ml_type = os.environ.get("ML_TYPE")
# print(ml_type)

def log_regression_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    rsq = r2_score(actual, pred)
    return {"rmse" : rmse, "mse" : mse, "Rsq" : rsq}

def log_classification_metrics(actual, pred):
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    # auc_score = auc(actual, pred)
    accuracy = accuracy_score(actual, pred)
    # return {"auc" : auc_score, "Accuracy" : accuracy}
    return {"Accuracy" : accuracy}

def generic_model(model_name, X_train, y_train, params = None):
    "What if we want multiple models?"
    pass

## XGBOOOST

@app.task
def xg_wo_loop(X_data, y_data, params, X_test, y_test):
    print("XG start " + str(params))
    with mlflow.start_run(run_name="ind_xg"):
        X_train = pd.read_json(X_data)
        y_train = pd.read_json(y_data, typ="series")
        X_test = pd.read_json(X_test)
        y_test = pd.read_json(y_test, typ="series")
        if ml_type == 0:
            xgb_model = xgb.XGBRegressor(**params)
        else:
            xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], early_stopping_rounds=50, verbose=0)
        pred = xgb_model.predict(X_train)
        if ml_type == 0:
            log_metrics(log_regression_metrics(y_train, pred))
        else:
            proba = xgb_model.predict_proba(X_train)
            log_metrics(log_classification_metrics(y_train, pred))
        log_params(params)

    print("XG completed " + str(params))

@app.task
def rf_wo_loop(X_data, y_data, params):
    print("RF start " + str(params))
    with mlflow.start_run(run_name="RF"):
        x_train = pd.read_json(X_data)
        y_train = pd.read_json(y_data, typ="series")
        # X_test = pd.read_json(X_test)
        # y_test = pd.read_json(y_test, typ="series")
        if ml_type == 0:
            rf_model = RandomForestRegressor(**params)
        else:
            rf_model = RandomForestClassifier(**params)
        rf_model.fit(x_train, y_train)
        pred = rf_model.predict(x_train)
        if ml_type == 0:
            log_metrics(log_regression_metrics(y_train, pred))
        else:
            proba = rf_model.predict_proba(x_train)
            log_classification_metrics(y_train, pred)
        log_params(params)
    print("RF completed " + str(params))

# xgparams = {"learning_rate" : [0.01, 0.1, 0.3], "n_estimators" : [100, 500, 1000]}
    
# @app.task
# def train_xgboost(X_data, y_data, params = xgparams):
#     """
#     Run XGBoost model
#     """
#     print("Starting xgboost")
#     unpack_params = ParameterGrid(params)

#     for i in unpack_params:
#         with mlflow.start_run(run_name='Xgboost'):
#             X_train = pd.read_json(X_data)
#             y_train = pd.read_json(y_data, typ="series")
#             if ml_type == 0:
#                 xgb_model = xgb.XGBRegressor(**i)
#             else:
#                 xgb_model = xgb.XGBClassifier(**i)
#             #modelling.apply_async((xgb_model, X_train, y_train, i), serializer='pickle')
#             xgb_model.fit(X_train, y_train)
#             pred = xgb_model.predict(X_train)
#             if ml_type == 0 :
#                 log_metrics(log_regression_metrics(y_train, pred))    
#             else:
#                 pass
#             log_params(i)
#     print("XGBoost Completed!")


## RANDOM FOREST

# rf_params = {"n_estimators":[100, 1000, 2000], "max_depth":[2,5,10]}    

# @app.task
# def train_rf(X_data, y_data, params=rf_params):
#     """
#     Run RandomForest
#     """
#     print("Starting rf")
#     unpack_params = ParameterGrid(params)
#     #rf = mlflow.set_experiment("rf")

#     for i in unpack_params:
#         with mlflow.start_run(run_name='RF'):
#             X_train = pd.read_json(X_data)
#             y_train = pd.read_json(y_data, typ="series")
#             rf_model = RandomForestRegressor(**i)
#             # modelling.apply_async((rf_model, X_train, y_train, i), serializer='pickle')
#             rf_model.fit(X_train, y_train)
#             pred = rf_model.predict(X_train)
#             log_metrics(log_regression_metrics(y_train, pred))
#             log_params(i)
#     print("RF Completed!")

## Linear Regression and Logistic Regression

# @app.task
# def linear_model(X_data, y_data):
#     """
#     Linear & Logistic Regression
#     """
#     print("Linear Model Starting!")
#     with mlflow.start_run(run_name='LM'):
#         x_train = pd.read_json(X_data)
#         y_train = pd.read_json(y_data, typ="series")
#         lm_model = LinearRegression()
#         lm_model.fit(x_train, y_train)
#         pred = lm_model.predict(x_train)
#         log_metrics(log_regression_metrics(y_train, pred))
#     print("Linear Model completed!")

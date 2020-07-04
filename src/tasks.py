from .celery import app
import pandas as pd
import xgboost as xgb
import os
import joblib

print('from tasks.py!')

@app.task
def train_xgbsoost(X_train, y_train, basedir):
    X_train = pd.read_json(X_train)
    y_train = pd.read_json(y_train, typ="series")
    xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators = 10) # scale_pos_weight = weight, 
    xgb_model.fit(X_train, y_train, eval_metric = "rmse")
    joblib.dump(xgb_model, basedir + "xgb.sav")
    #print(xgb_model)

@app.task
def add(x, y):
    return x + y
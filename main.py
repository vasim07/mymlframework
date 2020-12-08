import os
import pandas as pd
# from .train.xgboost import train_xgbsoost
from src.engines import  xg_wo_loop, rf_wo_loop#, linear_model
# from src.engines import train_xgboost, train_rf, linear_model, xg_wo_loop
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

base_dir = os.environ.get("BASE_DIR")
dataset = os.environ.get("TRAINING_DATA")
# X_test = os.environ.get("TRAINING_DATA")
# y_test = os.environ.get("TEST_DATA")
ml_type = os.environ.get("ML_TYPE")

X_data = pd.read_csv(dataset)

X = pd.get_dummies(X_data.drop(["Target"], axis = 1))
y = X_data["Target"]

# TODo - Stratify change based on ml_type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, train_size = 0.75, stratify = X_data["Target"], random_state = 42)

X_train = X_train.to_json()
X_test = X_test.to_json()
y_train = y_train.to_json()
y_test = y_test.to_json()

xg_params = {"learning_rate" : [0.01, 0.1, 0.3], "n_estimators" : [100, 500, 1000]}
unpack_xg = ParameterGrid(xg_params)
[xg_wo_loop.delay(X_train, y_train, i, X_test, y_test) for i in unpack_xg]
#train_xgboost.delay(X_train, y_train)

rf_params = {"n_estimators":[100, 1000, 2000], "max_depth":[2,5,10]}
unpack_rf = ParameterGrid(rf_params)
[rf_wo_loop.delay(X_train, y_train, i) for i in unpack_rf]

#train_rf.delay(X_train, y_train)

# linear_model.delay(X_train, y_train)

print("Check Celery")


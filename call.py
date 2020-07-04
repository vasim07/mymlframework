import os
import pandas as pd
# from .train.xgboost import train_xgbsoost
from src.tasks import train_xgbsoost, add

base_dir = os.environ.get("BASE_DIR")
dataset = os.environ.get("TRAINING_DATA")
# X_test = os.environ.get("TRAINING_DATA")
# y_test = os.environ.get("TEST_DATA")
# TYPE = os.environ.get("ML_TYPE")

X_train = pd.read_csv(dataset).drop(["Target"], axis = 1).to_json()
y_train = pd.read_csv(dataset)["Target"].to_json()

# print(X_train)
# print(y_train)
train_xgbsoost.delay(X_train, y_train, base_dir)

# add.delay(3, 4)

print("Working")


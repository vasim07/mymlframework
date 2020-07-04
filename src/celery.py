import os
from celery import Celery
import pandas as pd
import xgboost as xgb

app = Celery('downloaderApp',
             backend='redis://localhost:6379/1',
             broker='redis://localhost:6379/0', 
             include=['src.tasks'])


if __name__ == '__main__':
    app.start()
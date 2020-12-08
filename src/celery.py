import os
from celery import Celery
import pandas as pd
import xgboost as xgb

app = Celery('mlFrameworkApp',
             backend='redis://localhost:6379/1',
             broker='redis://localhost:6379/0', 
             include=['src.engines']
             )

# app.conf.update(CELERY_ACCEPT_CONTENT = ['pickle', 'json'])

if __name__ == '__main__':
    app.start()
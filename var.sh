export BASE_DIR=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/

export TRAINING_DATA=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/input/nb_data.csv
export TEST_DATA=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/input/nb_data.csv
# 0 - regression or 1 - classification
export ML_TYPE=0

rm -rf mlruns

# rm -rf directoryname

#export xgparams = {"learning_rate" : [0.01, 0.1, 0.3], "n_estimators" : [100, 500, 1000]}
#export rf_params = {"n_estimators":[100, 1000, 2000], "max_depth":[2,5,10]} 
# First install the module
# python -m src.predict

python /mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/main.py

# sudo service redis-server start
# sudo /etc/init.d/celeryd start/stop/status <- manually start celery
# celery -A src worker --loglevel=INFO
# flower -A src --port=5555
# Final - ./var.sh


# DOCKER 

# docker run --name mlframework01 --publish 5555:5555 --publish 5000:5000 -v f:/Vasim/PythonStuff/ml-framework/cdmlfw/input:/mnt/mlframework -it ubuntu /bin/bash
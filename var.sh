export BASE_DIR=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/

export TRAINING_DATA=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/src/input/nb_data.csv
export TEST_DATA=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/src/input/nb_data.csv
# regression or classification
export ML_TYPE=regression

# export MODEL=$1

# sudo service redis-server start
# sudo /etc/init.d/celeryd start/stop/status <- manually start celery
# celery -A src worker --loglevel=INFO
# flower -A src --port=5555

# First install the module
# python -m src.predict

python /mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/call.py

# Final - ./var.sh
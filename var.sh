export BASE_DIR=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/

export TRAINING_DATA=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/src/input/nb_data.csv
export TEST_DATA=/mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/src/input/nb_data.csv
# regression or classification
export ML_TYPE=regression

export MODEL=$1

# celery -A src worker --loglevel=INFO
# celery -A src worker --loglevel=DEBUG

# First install the module
#python -m src.predict

python /mnt/f/Vasim/PythonStuff/ml-framework/cdmlfw/call.py
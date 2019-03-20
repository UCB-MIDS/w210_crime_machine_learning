import numpy as np
import pandas as pd
import s3fs
import sys
import os
import tempfile
import pickle
import math
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

print('[' + str(datetime.now()) + '] Reading training dataset...')
sys.stdout.flush()
s3fs.S3FileSystem.read_timeout = 5184000  # one day
s3fs.S3FileSystem.connect_timeout = 5184000  # one day
try:
    #file = './data/OneHotEncodedDataset.parquet'                     # This line to read from local disk
    file = 's3://w210policedata/datasets/OneHotEncodedDataset.parquet'  # This line to read from S3
    #training_data = pd.read_csv(file,sep=',', error_bad_lines=False, dtype='unicode')
    training_data = pd.read_parquet(file)
except Exception as e:
    print('[' + str(datetime.now()) + '] Error reading input dataset: '+file)
    print('[' + str(datetime.now()) + '] Error message: '+str(e))
    print('[' + str(datetime.now()) + '] Aborting...')
    sys.exit(1)

print('[' + str(datetime.now()) + '] Training model...')
sys.stdout.flush()
df_Y = training_data.iloc[:,0]
df_X = training_data.iloc[:,1:]

### LINES BELOW FOR XGBREGRESSOR MODEL
trainval_X, test_X, trainval_y, test_y = train_test_split(df_X, df_Y, test_size = 0.10)
train_X, val_X, train_y, val_y = train_test_split(trainval_X, trainval_y, test_size = 0.20)
model = XGBRegressor(n_jobs=6)
model.fit(train_X,train_y, eval_set=[(val_X,val_y)], eval_metric='mae', verbose=True)
print('[' + str(datetime.now()) + '] Training complete!')
sys.stdout.flush()
print('[' + str(datetime.now()) + '] Running validation test...')
sys.stdout.flush()
XGBpredictions = model.predict(test_X)
MAE = mean_absolute_error(test_y , XGBpredictions)
MSE = mean_squared_error(test_y, XGBpredictions)
print('[' + str(datetime.now()) + ']    - XGBoost validation MAE = ',MAE)
print('[' + str(datetime.now()) + ']    - XGBoost validation MSE = ',MSE)
print('[' + str(datetime.now()) + ']    - XGBoost validation RMSE = ',math.sqrt(MSE))

print('[' + str(datetime.now()) + '] Persisting XGBoost model...')
sys.stdout.flush()
try:
    s3 = s3fs.S3FileSystem(anon=False)
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    model_file = "w210policedata/models/xgbregressor_model.joblib"
    joblib.dump(model,temp_file.name)
    s3.put(temp_file.name,model_file)
    temp_file.close()
except:
    print('[' + str(datetime.now()) + '] Error persisting trained model.')
    print('[' + str(datetime.now()) + '] Aborting...')
    sys.exit(1)
sys.stdout.flush()
### END OF XGBREGRESSOR MODEL

print('[' + str(datetime.now()) + '] Finished!')
sys.exit(0)

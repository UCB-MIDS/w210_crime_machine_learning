import argparse

parser = argparse.ArgumentParser(description='Trains an XGBoost Regressor machine learning model from selected features.')
parser.add_argument('modelname', metavar='modelname', type=str, help='Name for the model to be trained')
parser.add_argument('featurelist', metavar='features', nargs='+', type=str, help='List of features used on model training')
parser.add_argument('--list-features', action='store_true', default=False, dest='list_features', help='List available features')
args = parser.parse_args()

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

print('[' + str(datetime.now()) + '] Reading available features file...')
sys.stdout.flush()
try:
    #file = './data/OneHotEncodedDataset.parquet'                     # This line to read from local disk
    features_file = 's3://w210policedata/datasets/AvailableFeatures.pickle'  # This line to read from S3
    #training_data = pd.read_csv(file,sep=',', error_bad_lines=False, dtype='unicode')
    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(features_file, "rb") as json_file:
        available_features = pickle.load(json_file)
        json_file.close()
except Exception as e:
    print('[' + str(datetime.now()) + '] Error reading available features file: '+features_file)
    print('[' + str(datetime.now()) + '] Error message: '+str(e))
    print('[' + str(datetime.now()) + '] Aborting...')
    sys.exit(1)

if args.list_features:
    print('[' + str(datetime.now()) + '] Available features: ')
    for feature in available_features:
        if feature['ethnically_biased']:
            biased = 'Ethnically Biased'
        else:
            biased = 'Ethnically Unbiased'
        if feature['optional']:
            optional = 'Optional'
        else:
            optional = 'Required'
        if feature['onehot-encoded']:
            onehot = 'Categorical'
        else:
            onehot = 'Numerical'
        print('[' + str(datetime.now()) + ']    * '+feature['feature']+' ('+optional+', '+onehot+', '+biased+')')
    print('[' + str(datetime.now()) + '] Exiting...')
    sys.exit(0)

modelname = args.modelname
featurelist = args.featurelist
columns = []
selected_features = []
for feature in featurelist:
    exists = False
    for f in available_features:
        if f['feature'] == feature:
            exists = True
            if not (f['column'] in columns):
                columns.append(f['column'])
                selected_features.append(f)
            break
    if not exists:
        print('[' + str(datetime.now()) + '] Selected feature '+feature+' is not a valid available feature.')
        print('[' + str(datetime.now()) + '] Aborting...')
        sys.exit(1)

# Add required features
for f in available_features:
    if not f['optional'] and not (f['column'] in columns):
        columns.append(f['column'])
        selected_features.append(f)

print('[' + str(datetime.now()) + '] Training XGBoost Regressor model '+modelname+' with features:')
for feature in featurelist:
    print('[' + str(datetime.now()) + ']    * '+feature)
sys.stdout.flush()

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
# Pick only selected features for df_X
regex=""
for col in columns:
    regex += '('+col+')|'
regex = regex[:-1]
df_X = df_X.filter(regex=regex,axis=1)

### LINES BELOW FOR XGBREGRESSOR MODEL
trainval_X, test_X, trainval_y, test_y = train_test_split(df_X, df_Y, test_size = 0.10)
y_scaler = MinMaxScaler()
y_scaler.fit(trainval_y.values.reshape(-1,1))
trainval_y = y_scaler.transform(trainval_y.values.reshape(-1,1))
x_scaler = MinMaxScaler()
x_scaler.fit(trainval_X)
trainval_X = x_scaler.transform(trainval_X)
scaler = {'x':x_scaler,'y':y_scaler}
train_X, val_X, train_y, val_y = train_test_split(trainval_X, trainval_y, test_size = 0.20)
model = XGBRegressor(n_jobs=6)
model.fit(train_X,train_y, eval_set=[(val_X,val_y)], eval_metric='mae', verbose=True)
print('[' + str(datetime.now()) + '] Training complete!')
sys.stdout.flush()
print('[' + str(datetime.now()) + '] Running validation test...')
sys.stdout.flush()
test_X = scaler['x'].transform(test_X)
XGBpredictions = model.predict(test_X)
XGBpredictions = scaler['y'].inverse_transform(XGBpredictions.reshape(-1,1))
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
    model_file = "w210policedata/models/"+modelname+"/xgbregressor_model.joblib"
    joblib.dump(model,temp_file.name)
    s3.put(temp_file.name,model_file)
    temp_file.close()
    print('[' + str(datetime.now()) + '] Persisting model features...')
    sys.stdout.flush()
    features = df_X.columns.values.tolist()
    features_file = "w210policedata/models/"+modelname+"/xgbregressor_features.pickle"
    with s3.open(features_file, "wb") as json_file:
        pickle.dump(features, json_file, protocol=pickle.HIGHEST_PROTOCOL)
        json_file.close()
    print('[' + str(datetime.now()) + '] Persisting model scalers...')
    sys.stdout.flush()
    scaler_file = "w210policedata/models/"+modelname+"/xgbregressor_scaler.pickle"
    with s3.open(scaler_file, "wb") as json_file:
        pickle.dump(scaler, json_file, protocol=pickle.HIGHEST_PROTOCOL)
        json_file.close()
    print('[' + str(datetime.now()) + '] Persisting model information...')
    modelinfo = {'modelname': modelname,'type':'xgboost','features':selected_features,'statistics':{'MAE':MAE,'MSE':MSE,'RMSE':math.sqrt(MSE)}}
    info_file = "w210policedata/models/"+modelname+"/modelinfo.pickle"
    with s3.open(info_file, "wb") as json_file:
        pickle.dump(modelinfo, json_file, protocol=pickle.HIGHEST_PROTOCOL)
        json_file.close()
except:
    print('[' + str(datetime.now()) + '] Error persisting model assets.')
    print('[' + str(datetime.now()) + '] Aborting...')
    sys.exit(1)
sys.stdout.flush()
### END OF XGBREGRESSOR MODEL

print('[' + str(datetime.now()) + '] Finished!')
sys.exit(0)

import numpy as np
import pandas as pd
import s3fs
import sys
import os
import tempfile
import pickle
import math
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# print('[' + str(datetime.now()) + '] Performing some transformations...')
# sys.stdout.flush()
# training_data['Year'] = pd.to_datetime(training_data['Date']).dt.year
# print('[' + str(datetime.now()) + ']      * Dropping index column...')
# sys.stdout.flush()
# training_data.drop(columns=['Unnamed: 0'], inplace=True)
# print('[' + str(datetime.now()) + ']      * Consolidating data...')
# sys.stdout.flush()
# training_data = training_data.groupby(['Year','Primary Type','Beat','Weekday','Week of Year','Hour of the Day'])\
#                              .size().unstack().fillna(0).stack().reset_index(name='counts')
# print('[' + str(datetime.now()) + ']      * One-Hot Encoding categorical variables...')
# sys.stdout.flush()
# training_data = pd.concat([training_data,pd.get_dummies(training_data['Primary Type'], prefix='type')],axis=1)
# print('[' + str(datetime.now()) + ']          - Primary Type complete')
# sys.stdout.flush()
# training_data = pd.concat([training_data,pd.get_dummies(training_data['Beat'], prefix='beat')],axis=1)
# print('[' + str(datetime.now()) + ']          - Beat complete')
# sys.stdout.flush()
# training_data = pd.concat([training_data,pd.get_dummies(training_data['Weekday'], prefix='weekday')],axis=1)
# print('[' + str(datetime.now()) + ']          - Weekday complete')
# sys.stdout.flush()
# training_data = pd.concat([training_data,pd.get_dummies(training_data['Week of Year'], prefix='weekyear')],axis=1)
# print('[' + str(datetime.now()) + ']          - Week of year complete')
# sys.stdout.flush()
# training_data = pd.concat([training_data,pd.get_dummies(training_data['Hour of the Day'], prefix='hourday')],axis=1)
# print('[' + str(datetime.now()) + ']          - Hour of day complete')
# sys.stdout.flush()
# print('[' + str(datetime.now()) + ']      * Dropping unused columns...')
# sys.stdout.flush()
# training_data.drop(columns=['Year','Primary Type','Beat','Weekday','Week of Year','Hour of the Day'], inplace=True)

print('[' + str(datetime.now()) + '] Training model...')
sys.stdout.flush()
df_Y = training_data.iloc[:,0]
df_X = training_data.iloc[:,1:]
train_X, test_X, train_y, test_y = train_test_split(df_X, df_Y, test_size = 0.10)
y_scaler = MinMaxScaler()
y_scaler.fit(train_y.values.reshape(-1,1))
train_y = y_scaler.transform(train_y.values.reshape(-1,1))
x_scaler = MinMaxScaler()
x_scaler.fit(train_X)
train_X = x_scaler.transform(train_X)
scaler = {'x':x_scaler,'y':y_scaler}

### LINES BELOW FOR KERAS DEEP NEURAL NET MODEL
size_input = train_X.shape[1]
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)))
model = Sequential()
model.add(Dense(size_input, input_dim=size_input, kernel_initializer='normal', activation='relu'))
model.add(Dense(size_input*2, kernel_initializer='normal', activation='relu'))
#model.add(Dense(784, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
checkpoint_name = 'bestweights.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, verbose=1, mode='auto', restore_best_weights=True)
callbacks_list = [checkpoint,earlystop]
model.fit(train_X, train_y, epochs=500, batch_size=32768, validation_split = 0.2, callbacks=callbacks_list, verbose=2)
print('[' + str(datetime.now()) + '] Reloading best model checkpoint...')
model.load_weights("bestweights.hdf5")
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
print('[' + str(datetime.now()) + '] Persisting model structure...')
sys.stdout.flush()
try:
    s3 = s3fs.S3FileSystem(anon=False)
    struct_file = "w210policedata/models/keras_struct.json"
    with s3.open(struct_file, "w") as json_file:
        json_file.write(model.to_json())
        json_file.close()
    print('[' + str(datetime.now()) + '] Persisting model weights...')
    sys.stdout.flush()
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    weights_file = "w210policedata/models/keras_weights.h5"
    model.save_weights(temp_file.name)
    s3.put(temp_file.name,weights_file)
    temp_file.close()
    print('[' + str(datetime.now()) + '] Persisting model features...')
    sys.stdout.flush()
    features = df_X.columns.values.tolist()
    features_file = "w210policedata/models/keras_features.pickle"
    with s3.open(features_file, "wb") as json_file:
        pickle.dump(features, json_file, protocol=pickle.HIGHEST_PROTOCOL)
        json_file.close()
    print('[' + str(datetime.now()) + '] Persisting model scalers...')
    sys.stdout.flush()
    scaler_file = "w210policedata/models/keras_scaler.pickle"
    with s3.open(scaler_file, "wb") as json_file:
        pickle.dump(scaler, json_file, protocol=pickle.HIGHEST_PROTOCOL)
        json_file.close()
except:
    print('[' + str(datetime.now()) + '] Error persisting model assets.')
    print('[' + str(datetime.now()) + '] Aborting...')
    sys.exit(1)
### END OF KERAS DEEP NEURAL NET MODEL

### MODEL TESTING
print('[' + str(datetime.now()) + '] Training complete!')
sys.stdout.flush()
print('[' + str(datetime.now()) + '] Running validation test...')
sys.stdout.flush()
test_X = scaler['x'].transform(test_X)
predictions = model.predict(test_X)
predictions = scaler['y'].inverse_transform(predictions)
MAE = mean_absolute_error(test_y , predictions)
MSE = mean_squared_error(test_y, predictions)
print('[' + str(datetime.now()) + ']    - Keras Deep Neural Net validation MAE = ',MAE)
print('[' + str(datetime.now()) + ']    - Keras Deep Neural Net validation MSE = ',MSE)
print('[' + str(datetime.now()) + ']    - Keras Deep Neural Net validation RMSE = ',math.sqrt(MSE))

print('[' + str(datetime.now()) + '] Finished!')
sys.exit(0)

# W210 Police Deployment
# MACHINE LEARNING Microservices

import numpy as np
import pandas as pd
import subprocess
import shlex
import threading
import s3fs
import tempfile
import pickle
import joblib
import json
import itertools
import configparser
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

def load_keras_model():
    from keras.models import model_from_json
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import backend as K
    from keras.wrappers.scikit_learn import KerasRegressor
    import tensorflow as tf
    s3fs.S3FileSystem.read_timeout = 5184000  # one day
    s3fs.S3FileSystem.connect_timeout = 5184000  # one day
    s3 = s3fs.S3FileSystem(anon=False)
    K.clear_session()
    struct_file = 'w210policedata/models/keras_struct.json'
    weights_file = 'w210policedata/models/keras_weights.h5'
    features_file = 'w210policedata/models/keras_features.pickle'
    with s3.open(struct_file, "r") as json_file:
        model = model_from_json(json_file.read())
        json_file.close()
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    s3.get(weights_file,temp_file.name)
    model.load_weights(temp_file.name)
    graph = tf.get_default_graph()
    temp_file.close()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    with s3.open(features_file, "rb") as pickle_file:
        model_features = pickle.load(pickle_file)
        pickle_file.close()
    model_type = 'keras'
    return model,model_features,graph,model_type

def load_xgb_model():
    from xgboost import XGBRegressor
    s3fs.S3FileSystem.read_timeout = 5184000  # one day
    s3fs.S3FileSystem.connect_timeout = 5184000  # one day
    s3 = s3fs.S3FileSystem(anon=False)
    model_file = 'w210policedata/models/xgbregressor_model.joblib'
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    s3.get(model_file,temp_file.name)
    model = joblib.load(temp_file.name)
    graph = None
    temp_file.close()
    model_features = model.get_booster().feature_names
    model_type = 'xgboost'
    return model,model_features,graph,model_type

application = Flask(__name__)
api = Api(application)

application.config['CORS_ENABLED'] = True
CORS(application)

runningProcess = None
processStdout = []

model = None
model_features = None
model_type = None
graph = None

### Load default model configuration from configuration file
s3fs.S3FileSystem.read_timeout = 5184000  # one day
s3fs.S3FileSystem.connect_timeout = 5184000  # one day
s3 = s3fs.S3FileSystem(anon=False)
config_file = 'w210policedata/config/ml.ini'
try:
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    s3.get(config_file,temp_file.name)
    config = configparser.ConfigParser()
    config.read(temp_file.name)
except:
    print('Failed to load configuration file.')
    print('Creating new file with default values.')
    config = configparser.ConfigParser()
    config['GENERAL'] = {'DefaultModel': 'xgboost'}
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(temp_file.name, 'w') as confs:
        config.write(confs)
    s3.put(temp_file.name,config_file)
    temp_file.close()
default_model = config['GENERAL']['DefaultModel']

if default_model == 'keras':
    model,model_features,graph,model_type = load_keras_model()
else:
    model,model_features,graph,model_type = load_xgb_model()

# Services to implement:
#   * Train
#   * Predict
#   * Evaluate model

def processTracker(process):
    for line in iter(process.stdout.readline, b''):
        processStdout.append('{0}'.format(line.decode('utf-8')))
    process.poll()

class checkService(Resource):
    def get(self):
        # Test if the service is up
        return {'message':'Machine learning service is running.','result': 'success'}

class trainModel(Resource):
    def get(self):
        # Run background worker to read from S3, transform and write back to S3
        global runningProcess
        global processStdout

        trainParser = reqparse.RequestParser()
        trainParser.add_argument('modeltype')
        args = trainParser.parse_args()

        if args['modeltype'] is None:
            return {'message':'Missing modeltype argument. Supported types: keras, xgboost.','result':'failed'}


        if (runningProcess is not None):
            if (runningProcess.poll() is not None):
                return {'message':'There is a model training job currently running.','pid':runningProcess.pid,'result': 'failed'}
        try:
            if args['modeltype'] == 'keras':
                command = 'python trainer_keras.py'
            else:
                command = 'python trainer_xgbregressor.py'
            runningProcess = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            processStdout = []
            t = threading.Thread(target=processTracker, args=(runningProcess,))
            t.start()
        except:
            return{'message':'Model training failed.','pid':None,'result': 'failed'}
        return {'message':'Model training started.','pid':runningProcess.pid,'result': 'success'}

class getTrainingStatus(Resource):
    def get(self):
        # Check if the background worker is running and how much of the work is completed
        if (runningProcess is not None):
            returncode = runningProcess.poll()
            if (returncode is not None):
                if (returncode != 0):
                    return {'returncode':returncode,'status':'Model training failed','stdout':processStdout}
                else:
                    return {'returncode':returncode,'status':'Model training finished succesfully','stdout':processStdout}
            else:
                return {'returncode':None,'status':'Model is still training','stdout':processStdout}
        return {'returncode':None,'status':'No model train running','stdout': None}

class killTrainer(Resource):
    def get(self):
        # Check if the worker is running and kill it
        if (runningProcess is not None):
            returncode = runningProcess.poll()
            if (returncode is None):
                runningProcess.kill()
                return {'message':'Kill signal sent to model trainer.','result':'success'}
        return {'message':'No model training running','result': 'failed'}

class predict(Resource):
    # Get predictors
    def get(self):
        global model
        global model_features
        global model_type
        if (model is None):
            return {'message':'Model is not loaded','result':'failed'}
        return {'model_type':model_type,'input_features':model_features,'result':'success'}
    # Run the predictions
    def post(self):
        global model
        global model_features
        global graph
        global model_type

        predictParser = reqparse.RequestParser()
        predictParser.add_argument('communityarea')
        predictParser.add_argument('weekday')
        predictParser.add_argument('weekyear')
        predictParser.add_argument('hourday')

        if (model is None):
            return {'message':'Model is not loaded','result':'failed'}
        args = predictParser.parse_args()
        for arg in args:
            if args[arg] is None:
                if (arg == 'communityarea'):
                    args[arg] = [x.replace('communityArea_','') for x in model_features if x.startswith('communityArea_')]
                else:
                    return {'message':'Missing input '+arg,'result':'failed'}
            else:
                args[arg] = json.loads(args[arg])
        df = pd.DataFrame(columns=model_features)
        crime_types = [x for x in model_features if x.startswith('primaryType_')]
        results = []
        for ca,wy,wd,hd,ct in itertools.product(args['communityarea'],args['weekyear'],args['weekday'],args['hourday'],crime_types):
            line = {'communityArea_'+str(ca):1,'weekYear_'+str(wy):1,'weekDay_'+str(wd):1,'hourDay_'+str(hd):1,ct:1}
            df = df.append(line, ignore_index=True)
            results.append({'communityArea':str(ca),'weekYear':wy,'weekDay':wd,'hourDay':hd,'primaryType':ct.replace('primaryType_',''),'pred':None})
        df.fillna(0,inplace=True)
        if (model_type == 'keras'):
            with graph.as_default():
                prediction = model.predict(df)
        else:
            prediction = model.predict(df)
        print(prediction)
        for i in range(len(prediction)):
            if model_type == 'keras':
                results[i]['pred'] = int(np.round(float(prediction[i][0])-0.39+0.5))
            else:
                results[i]['pred'] = int(np.round(float(prediction[i])-0.39+0.5))
        return {'result':results}

class reloadModel(Resource):
    def get(self):
        # Reload the model
        global model
        global model_features
        global graph
        global model_type

        loadParser = reqparse.RequestParser()
        loadParser.add_argument('modeltype')
        args = loadParser.parse_args()

        if args['modeltype'] is None:
            return {'message':'Missing modeltype argument. Supported types: keras, xgboost.','result':'failed'}

        try:
            if args['modeltype'] == 'keras':
                model,model_features,graph,model_type = load_keras_model()
            else:
                model,model_features,graph,model_type = load_xgb_model()
            return{'message':'Model reloaded succesfully.','error':None,'result': 'success'}
        except Exception as e:
            return{'message':'Model load failed.','error':str(e),'result': 'failed'}

api.add_resource(checkService, '/')
api.add_resource(trainModel, '/trainModel')
api.add_resource(getTrainingStatus, '/getTrainingStatus')
api.add_resource(killTrainer, '/killTrainer')
api.add_resource(predict, '/predict')
api.add_resource(reloadModel, '/reloadModel')

if __name__ == '__main__':
    application.run(debug=True, port=60000)

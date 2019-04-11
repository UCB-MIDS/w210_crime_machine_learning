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
from scipy.stats import t
from collections import defaultdict
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

# Calculation of prediction Fairness
# Uses a difference of means test as described in https://link.springer.com/article/10.1007%2Fs10618-017-0506-1
def calculateFairness(communities, predictions):
    comm_count = {0: 0, 1: 0}
    predicted_count = {0: 0, 1: 0}

    for comm in predictions:
        comm_code = int(comm)
        if (communities[comm_code]['ethnicity'] == 0) or (communities[comm_code]['ethnicity'] == 1):
            comm_count[1] += 1
            predicted_count[1] += predictions[comm]
        else:
            comm_count[0] += 1
            predicted_count[0] += predictions[comm]

    df = comm_count[0]+comm_count[1]-2

    if (predicted_count[0] == 0) and (predicted_count[1] == 0):
        return 1

    means = {0: predicted_count[0]/comm_count[0], 1: predicted_count[1]/comm_count[1]}

    variances = {0: 0, 1: 0}

    for comm in predictions:
        comm_code = int(comm)
        if (communities[comm_code]['ethnicity'] == 0) or (communities[comm_code]['ethnicity'] == 1):
            variances[1] += (predictions[comm]-means[1])**2
        else:
            variances[0] += (predictions[comm]-means[0])**2

    variances = {0: variances[0]/(comm_count[0]-1), 1: variances[1]/(comm_count[1]-1)}

    sigma = ((((comm_count[0]-1)*(variances[0]**2))+((comm_count[1]-1)*(variances[1]**2)))/(comm_count[0]+comm_count[1]-2))**0.5

    t_stat = (means[0]-means[1])/(sigma*(((1/comm_count[0])+(1/comm_count[1]))**0.5))

    fairness = (1 - t.cdf(abs(t_stat), df)) * 2
    fairness = fairness*100

    return fairness

def load_keras_model(modelname):
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
    struct_file = 'w210policedata/models/'+modelname+'/keras_struct.json'
    weights_file = 'w210policedata/models/'+modelname+'/keras_weights.h5'
    features_file = 'w210policedata/models/'+modelname+'/keras_features.pickle'
    scaler_file = 'w210policedata/models/'+modelname+'/keras_scaler.pickle'
    modelinfo_file = 'w210policedata/models/'+modelname+'/modelinfo.pickle'
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
    with s3.open(scaler_file, "rb") as pickle_file:
        model_scalers = pickle.load(pickle_file)
        pickle_file.close()
    with s3.open(modelinfo_file, "rb") as pickle_file:
        model_info = pickle.load(pickle_file)
        pickle_file.close()
    model_name = model_info['modelname']
    return model,model_name,model_features,graph,model_type,model_scalers,model_info

def load_xgb_model(modelname):
    from xgboost import XGBRegressor
    s3fs.S3FileSystem.read_timeout = 5184000  # one day
    s3fs.S3FileSystem.connect_timeout = 5184000  # one day
    s3 = s3fs.S3FileSystem(anon=False)
    model_file = 'w210policedata/models/'+modelname+'/xgbregressor_model.joblib'
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    s3.get(model_file,temp_file.name)
    model = joblib.load(temp_file.name)
    graph = None
    temp_file.close()
    model_features = model.get_booster().feature_names
    model_type = 'xgboost'
    model_scalers = None
    model_info = None
    model_name=None
    return model,model_name,model_features,graph,model_type,model_scalers,model_info

def load_model(modelname):
    modelinfo_file = 'w210policedata/models/'+modelname+'/modelinfo.pickle'
    with s3.open(modelinfo_file, "rb") as pickle_file:
        model_info = pickle.load(pickle_file)
        pickle_file.close()
    if model_info['type'] == 'keras':
        return load_keras_model(modelname)
    else:
        return load_xgb_model(modelname)

### Load Flask configuration file
s3fs.S3FileSystem.read_timeout = 5184000  # one day
s3fs.S3FileSystem.connect_timeout = 5184000  # one day
s3 = s3fs.S3FileSystem(anon=False)
config_file = 'w210policedata/config/config.py'
try:
    s3.get(config_file,'config.py')
except:
    print('Failed to load application configuration file!')

application = Flask(__name__)
api = Api(application)
application.config.from_pyfile('config.py')
db = SQLAlchemy(application)

application.config['CORS_ENABLED'] = True
CORS(application)

## Define the DB model
class Community(db.Model):
    __tablename__ = 'community'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.Integer)
    name = db.Column(db.String(255))
    ethnicity = db.Column(db.Integer)

    def __str__(self):
        return self.name

runningProcess = None
processStdout = []

model = None
model_name = None
model_features = None
model_type = None
graph = None
model_scalers = None
model_info = None

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
    config['GENERAL'] = {'DefaultModel': 'keras'}
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(temp_file.name, 'w') as confs:
        config.write(confs)
    s3.put(temp_file.name,config_file)
    temp_file.close()
default_model = config['GENERAL']['DefaultModel']

model,model_name,model_features,graph,model_type,model_scalers,model_info = load_model(default_model)

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
        trainParser.add_argument('modelname')
        trainParser.add_argument('modeltype')
        trainParser.add_argument('features')
        args = trainParser.parse_args()

        if args['modelname'] is None:
            return {'message':'Missing modelname argument.','result':'failed'}
        if args['modeltype'] is None:
            return {'message':'Missing modeltype argument. Supported types: keras, xgboost.','result':'failed'}
        if args['features'] is None:
            return {'message':'Missing features argument.','result':'failed'}

        if (runningProcess is not None):
            if (runningProcess.poll() is not None):
                return {'message':'There is a model training job currently running.','pid':runningProcess.pid,'result': 'failed'}
        try:
            if json.loads(args['modeltype']) == 'keras':
                command = 'python trainer_keras.py'
            else:
                command = 'python trainer_xgbregressor.py'
            command += ' '+json.loads(args['modelname'])
            for feature in json.loads(args['features']):
                command += ' "'+feature+'"'
            print(shlex.split(command))
            runningProcess = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            processStdout = []
            t = threading.Thread(target=processTracker, args=(runningProcess,))
            t.start()
        except:
            return{'message':'Model training failed.','pid':None,'result': 'failed'}
        return {'message':'Model training started.','pid':runningProcess.pid,'result': 'success'}

class getTrainingStatus(Resource):
    def get(self):
        global runningProcess
        global processStdout

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
        return {'returncode':None,'status':'No model training running','stdout': None}

class killTrainer(Resource):
    def get(self):
        global runningProcess
        global processStdout

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
        global model_name
        global model_features
        global model_type
        if (model is None):
            return {'message':'Model is not loaded','result':'failed'}
        return {'model_name':model_name,'model_type':model_type,'input_features':model_features,'result':'success'}
    # Run the predictions
    def post(self):
        global model
        global model_features
        global graph
        global model_type
        global model_scalers
        global model_info

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
            df = model_scalers['x'].transform(df)
            with graph.as_default():
                prediction = model.predict(df)
                prediction = model_scalers['y'].inverse_transform(prediction)
        else:
            prediction = model.predict(df)
        for i in range(len(prediction)):
            if model_type == 'keras':
                results[i]['pred'] = int(max(np.round(float(prediction[i][0])-0.39+0.5),0))
            else:
                results[i]['pred'] = int(max(np.round(float(prediction[i])-0.39+0.5),0))
        return {'result':results}

class predictionAndKPIs(Resource):
    # Get predictors
    def get(self):
        global model
        global model_features
        global model_type
        global model_name
        if (model is None):
            return {'message':'Model is not loaded','result':'failed'}
        return {'model_name':model_name,'model_type':model_type,'input_features':model_features,'result':'success'}
    # Run the predictions
    def post(self):
        global model
        global model_features
        global graph
        global model_type
        global model_scalers
        global model_info

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
            df = model_scalers['x'].transform(df)
            with graph.as_default():
                prediction = model.predict(df)
                prediction = model_scalers['y'].inverse_transform(prediction)
        else:
            prediction = model.predict(df)
        for i in range(len(prediction)):
            if model_type == 'keras':
                results[i]['pred'] = int(max(np.round(float(prediction[i][0])-0.39+0.5),0))
            else:
                results[i]['pred'] = int(max(np.round(float(prediction[i])-0.39+0.5),0))

        # Consolidate into map format and calculate KPIs
        crimeByCommunity = defaultdict(int)
        crimeByType = defaultdict(int)
        communities = {}
        predictionFairness = 0

        for comm in db.session.query(Community):
            communities[comm.code] = {'id':comm.id,'code':comm.code,'name':comm.name,'ethnicity':comm.ethnicity}

        for result in results:
            if (result['communityArea'] is not None) and (result['primaryType'] is not None) and (result['communityArea'] != '0') and (result['primaryType'] != ''):
                crimeByCommunity[result['communityArea']] += result['pred']
                crimeByType[result['primaryType']] += result['pred']

        predictionFairness = calculateFairness(communities,crimeByCommunity)

        return {'crimeByCommunity':crimeByCommunity, 'crimeByType':crimeByType, 'fairness': predictionFairness, 'predictions':results, 'result':'success'}

class reloadModel(Resource):
    def get(self):
        # Reload the model
        global model
        global model_name
        global model_features
        global graph
        global model_type
        global model_scalers
        global model_info

        loadParser = reqparse.RequestParser()
        loadParser.add_argument('modelname')
        args = loadParser.parse_args()

        if args['modelname'] is None:
            return {'message':'Missing modelname argument.','result':'failed'}

        try:
            model,model_name,model_features,graph,model_type,model_scalers,model_info = load_model(json.loads(args['modelname']))
            return{'message':'Model loaded succesfully.','error':None,'result': 'success'}
        except Exception as e:
            return{'message':'Model load failed.','error':str(e),'result': 'failed'}

class getAvailableModels(Resource):
    def get(self):
        # Look into S3 Models folder for trained models
        models = []
        try:
            s3 = s3fs.S3FileSystem(anon=False)
            items = s3.ls('w210policedata/models',detail=True)
            for item in items:
                if item['StorageClass'] == 'DIRECTORY':
                    modelinfo_file = item['Key']+'/modelinfo.pickle'
                    with s3.open(modelinfo_file, "rb") as pickle_file:
                        model_info = pickle.load(pickle_file)
                        pickle_file.close()
                    models.append(model_info)
        except Exception as e:
            return{'message':'Failure reading model data from S3.','error':str(e),'result':'failed'}
        return {'models':models,'result':'success'}

class getAvailableFeatures(Resource):
    def get(self):
        try:
            #file = './data/OneHotEncodedDataset.parquet'                     # This line to read from local disk
            features_file = 's3://w210policedata/datasets/AvailableFeatures.pickle'  # This line to read from S3
            #training_data = pd.read_csv(file,sep=',', error_bad_lines=False, dtype='unicode')
            s3 = s3fs.S3FileSystem(anon=False)
            with s3.open(features_file, "rb") as json_file:
                available_features = pickle.load(json_file)
                json_file.close()
        except Exception as e:
            return{'message':'Failure reading available features data from S3.','error':str(e),'result':'failed'}
        return {'features':available_features,'result':'success'}

api.add_resource(checkService, '/')
api.add_resource(trainModel, '/trainModel')
api.add_resource(getTrainingStatus, '/getTrainingStatus')
api.add_resource(killTrainer, '/killTrainer')
api.add_resource(predict, '/predict')
api.add_resource(predictionAndKPIs, '/predictionAndKPIs')
api.add_resource(reloadModel, '/reloadModel')
api.add_resource(getAvailableModels, '/getAvailableModels')
api.add_resource(getAvailableFeatures, '/getAvailableFeatures')

if __name__ == '__main__':
    application.run(debug=True, port=60000)

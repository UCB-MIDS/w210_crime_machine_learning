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
from flask import Flask
from flask_restful import Resource, Api
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from keras.wrappers.scikit_learn import KerasRegressor

application = Flask(__name__)
api = Api(application)

runningProcess = None
processStdout = []

model = None
model_features = None

s3fs.S3FileSystem.read_timeout = 5184000  # one day
s3fs.S3FileSystem.connect_timeout = 5184000  # one day
s3 = s3fs.S3FileSystem(anon=False)
struct_file = 'w210policedata/models/keras_struct.json'
weights_file = 'w210policedata/models/keras_weights.h5'
features_file = 'w210policedata/models/keras_features.pickle'
try:
    with s3.open(struct_file, "r") as json_file:
        model = model_from_json(json_file.read())
        json_file.close()
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    s3.get(weights_file,temp_file.name)
    model.load_weights(temp_file.name)
    temp_file.close()
    with s3.open(features_file, "rb") as pickle_file:
        model_features = pickle.load(pickle_file).values.tolist()
        pickle_file.close()
except Exception as e:
    print('Model could not be loaded!')
    print(e)

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
        if (runningProcess is not None):
            if (runningProcess.poll() is not None):
                return {'message':'There is a model training job currently running.','pid':runningProcess.pid,'result': 'failed'}
        try:
            command = 'python trainer.py'
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
        if (model is None):
            return {'message':'Model is not loaded','result':'failed'}
        return {'input_features':model_features,'result':'success'}
    # Run the predictions
    def post(self,data):
        return {'result':data}

class reloadModel(Resource):
    def get(self):
        # Reload the model
        global model
        global model_features
        global s3
        global struct_file
        global weights_file
        global features_file
        # Reset model
        backend.clear_session()
        model = None
        model_features = None
        try:
            with s3.open(struct_file, "r") as json_file:
                model = model_from_json(json_file.read())
                json_file.close()
            temp_file = tempfile.NamedTemporaryFile(delete=True)
            s3.get(weights_file,temp_file.name)
            model.load_weights(temp_file.name)
            temp_file.close()
            with s3.open(features_file, "rb") as pickle_file:
                model_features = pickle.load(pickle_file).values.tolist()
                pickle_file.close()
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
    application.run(debug=True)

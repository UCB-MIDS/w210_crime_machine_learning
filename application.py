# W210 Police Deployment
# MACHINE LEARNING Microservices

import numpy as np
import pandas as pd
import subprocess
import shlex
import threading
from flask import Flask
from flask_restful import Resource, Api

application = Flask(__name__)
api = Api(application)

runningTraining = None
trainingStdout = []

runningPrediction = None
predictionStdout = []

def processTracker(process):
    for line in iter(process.stdout.readline, b''):
        processStdout.append('{0}'.format(line.decode('utf-8')))
    process.poll()

class checkService(Resource):
    def get(self):
        # Test if the service is up
        return {'message':'Machine learning service is running.','result': 'success'}

class runJob(Resource):
    def get(self):
        # Run background worker to read from S3, transform and write back to S3
        global runningProcess
        global processStdout
        if (runningProcess is not None):
            if (runningProcess.poll() is not None):
                return {'message':'There is a job currently running.','pid':runningProcess.pid,'result': 'failed'}
        try:
            #command = 'python test.py'
            command = 'python worker.py'
            runningProcess = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            processStdout = []
            t = threading.Thread(target=processTracker, args=(runningProcess,))
            t.start()
        except:
            return{'message':'Job execution failed.','pid':None,'result': 'failed'}
        return {'message':'Job execution started.','pid':runningProcess.pid,'result': 'success'}


class getJobStatus(Resource):
    def get(self):
        # Check if the background worker is running and how much of the work is completed
        if (runningProcess is not None):
            returncode = runningProcess.poll()
            if (returncode is not None):
                if (returncode != 0):
                    return {'returncode':returncode,'status':'Job failed','stdout':processStdout}
                else:
                    return {'returncode':returncode,'status':'Job executed succesfully','stdout':processStdout}
            else:
                return {'returncode':None,'status':'Job is still running','stdout':processStdout}
        return {'returncode':None,'status':'No job running','stdout': None}

class killJob(Resource):
    def get(self):
        # Check if the worker is running and kill it
        if (runningProcess is not None):
            returncode = runningProcess.poll()
            if (returncode is None):
                runningProcess.kill()
                return {'message':'Kill signal sent to job.','result':'success'}
        return {'message':'No job running','result': 'failed'}

api.add_resource(checkService, '/')
api.add_resource(runJob, '/runJob')
api.add_resource(getJobStatus, '/getJobStatus')
api.add_resource(killJob, '/killJob')

if __name__ == '__main__':
    application.run(debug=True)

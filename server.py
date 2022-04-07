from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify
from flask import request

from joblib import load
import numpy as np
import pandas as pd
import autosklearn.classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

import plotly.express as px
import plotly.graph_objects as go
set_config(display='diagram')

mycolumns = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/", methods=['POST'])

model_path = "/content/drive/MyDrive/content/drive/MyDrive/DS/exercises/ASSGNMENT/models/model2021-04-18_07:51:07.329098.pkl"

def home():
    automl = load(model_path)
    json_data = request.get_json()
    value = list(automl.predict(pd.DataFrame([json_data['features']], columns=mycolumns)))
    return jsonify(prediction=int(value[0]), json=True)
  
app.run()
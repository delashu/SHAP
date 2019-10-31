########################################################################################################
########################################################################################################
########################################################################################################
####	MAKE SURE THE DIRECTORY THAT THE FILE IS DOWNLOADED TO IS THE SAME THAT THE PROGRAM READS 	####
####	 SEE https://github.com/Kaggle/kaggle-api FOR DOCUMENTATION FOR DOWNLOADING VIA THE API 	####
####																								####
####  The file can be downloaded manually at: https://www.kaggle.com/harlfoxem/housesalesprediction	####
####																								####
####	  https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d		####
########################################################################################################
########################################################################################################
########################################################################################################

import subprocess
import os.path

#download file if it doesn't already exist (may need to specify path) 
if not (os.path.isfile("~\\kc_house_data.csv")):
	#may need to specify download path with "-p [filepath]"
	subprocess.Popen('kaggle datasets download -d harlfoxem/housesalesprediction --unzip')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

np.random.seed(0)
def shap_plot(j):
    explainerModel = shap.TreeExplainer(model)
    shap_values_Model = explainerModel.shap_values(S)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]], matplotlib=True)
    return(p)

#read file in from location downloaded
df = pd.read_csv('kc_house_data.csv') 

# The target variable is 'price'.
Y = df['price']
X =  df[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']]

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Build the model with random forest regression:
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)

# Create SHAP values
shap_values = shap.TreeExplainer(model).shap_values(X_train)

# Create overall summary plot (bar chart without individual observations)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Create overall summary plot (with individual observations)
shap.summary_plot(shap_values, X_train)

# Create partial dependence plot for variable "long"
shap.dependence_plot("long", shap_values, X_train)

# Create dataset to set up individual shap value plots
X_output = X_test.copy()
X_output.loc[:,'predict'] = np.round(model.predict(X_output),2)

# Randomly pick observations from test dataset
random_picks = np.arange(1,len(X_output),50)
S = X_output.iloc[random_picks]

# Create local explanation plot for data from S using function above 
shap_plot(0)

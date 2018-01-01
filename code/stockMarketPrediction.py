###########################################################################
############  Stock-Market-Prediction_IBEX35_TENSORFLOW ###################
#######################################################################
###############               Authors:                      ###############  
######			       Juan Manuel López Torralba                    ######
######                                                               ###### 
######  OpenSource with Creative Commons Attribution-NonCommercial-  ######
######  ShareAlike license ubicated in GitHub throughout:            ######
## https://github.com/jmlopezt/Stock-Market-Prediction_IBEX35_TENSORFLOW ##
######                                                               ######
######  This work is licensed under the Creative Commons             ######
######  Attribution-NonCommercial-ShareAlike CC BY-NC-SA License.    ######
######  To view a copy of the license, visit                         ######
######  https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode  ######
###############                                             ###############
###########################################################################
###########################################################################

# Import modules

# Tensorflow
import tensorflow as tensorflow

# Numpy arrays
import numpy as np

# Read data from CSV file
import pandas as pd

# Used to convert date string to numerical value
from datetime import datetime, timedelta

# Used to plot data
import matplotlib.pyplot as mpl

# Csv data

dataFrame = pd.read_csv('csvfiles/ibex35/ibex35.csv')
#data = pd.read_csv('csvfiles/ibex35/endesa.csv')
#data = pd.read_csv('csvfiles/ibex35/bbva.csv')
#data = pd.read_csv('csvfiles/ibex35/mapfre.csv')

# Parse data

dateAux = dataFrame['Date'].values
dateTimeAux = np.zeros(dateAux.shape)

	#date strings to numeric value
for i, j in enumerate(dateAux):

	dateTimeAux[i] = datetime.strptime(j, '%Y-%m-%d').timestamp()
    #Add the newly parsed column to the dataframe
	#dataFrame['Timestamp'] = dateTimeAux
    Timestamp = dateTimeAux

#Remove any unused columns (axis = 1 specifies fields are columns)
dataFrame = dataFrame.drop(['Date','Open','High','Low','Adj Close','Volume'],axis=1)
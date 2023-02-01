# models for predict the gols
'''
    1 option --> Predict the total gols (Regresion)
    features --> DTO, Time, HomeTeam, AwayTeam, Referee
    
    2 option --> Predict if total gols +- 2.5 (Classifier)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets_the_goals import df_2

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, recall_score, confusion_matrix

def label_encoder(data):
    data[['Time','HomeTeam','AwayTeam','Referee']] = data[['Time','HomeTeam','AwayTeam','Referee']].apply(LabelEncoder().fit_transform) 
    return data

def split_the_data(data, test_size):
    x = data.drop('TG', axis=1)
    y = data['TG']
    return train_test_split(x, y, test_size=test_size, random_state=10)

def standard_x(x_train, x_test):
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
def predict(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

def metrics(y_test, y_pred):
    mse = np.sqrt(mean_squared_error(y_pred, y_test))
    abs = mean_absolute_error(y_pred, y_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    confusion_matrix_ = confusion_matrix(y_test, y_pred)

opa = np.array([10,0,13,0,2.5,2.5,2.333,5.333]).reshape(-1,1)

df_1_LE = label_encoder(df_2)
x_train, x_test, y_train, y_test = split_the_data(df_1_LE, 0.25)
standard_x(x_train, x_test)
predict(DecisionTreeClassifier(max_depth=4), x_train, x_test, y_train, y_test)

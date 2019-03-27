import os
os.chdir('/Users/nmace/Desktop/IA')  

import pandas as pd
import numpy as np
import tensorflow as tf
from IA_Process import DataProcessing
import matplotlib.pyplot as plt


data= pd.read_csv("d_historique.txt", header=0, delimiter="\t")

def train_model_LSTM(data,asset_name):
    
    df=data.loc[:,['Date',asset_name]]
    process = DataProcessing(df, 0.9)
    process.gen_test(10)
    process.gen_train(10)
    X_train = process.X_train.reshape((219, 10,1))
    Y_train = process.Y_train
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20,input_shape=(10, 1), return_sequences=True))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, Y_train, epochs=100)
    return model


def forecast_next_value_LSTM(data,jour,asset_name, model):
   
    stock=data.iloc[jour-10:jour][asset_name]
    X_predict = np.array(stock).reshape((1, 10, 1))
    out=model.predict(X_predict)
    return(out[0][0])

asset_name = "EURGBP_SPOT"
index = 150
model=train_model_LSTM(data,asset_name)
forecast = forecast_next_value_LSTM(data, index,asset_name,model)
df = data.loc[:,['Date',asset_name]]

plt.plot(df.EURGBP_SPOT[index-41:index+2])
plt.scatter(index+1,forecast)
print(forecast)


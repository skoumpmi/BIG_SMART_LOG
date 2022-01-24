#from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone, timedelta
# multivariate multi-step data preparation
from numpy import array
from numpy import hstack
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
#import make_normalizations_new
from tensorflow import keras
import pickle
from multiprocessing import Process

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate, Flatten, Multiply, Activation, \
    Add
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import configparser
import os
from os import path
from pathlib import Path
from keras import backend
from sklearn.preprocessing import MinMaxScaler
import statistics
from pickle import load
from configobj import ConfigObj
from flask import Flask, request, jsonify, json, render_template,redirect, url_for
config_file = 'config.ini'
config = ConfigObj(config_file,list_values=True,interpolation=True)
sections = config.keys()

app = Flask(__name__)
@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello from self-learning app!!</h1>"
@app.route("/TrainWavenet", methods=['POST'])
def start_training():
    request_body = request.get_json()
    print(request_body)
    print(list(request_body[0].values()))
    print(type(request_body[0]))
   
    sections = config.keys()
    warehouse = list(request_body[0].values())[1]
    target = list(request_body[0].values())[0]
    n_filters = int(config[target][warehouse ]['n_filters'])
    filter_width = int(config[target][warehouse ]['filter_width'])
    batch_size = int(config[target][warehouse ]['batch_size'])
    windowsize = int(config[target][warehouse ]['windows_size'])
    activation_function = config[target][warehouse ]['activation_function']
    timestep = int(config[target][warehouse ]['timestep'])
    numepochs = 200
    print(warehouse)
    print(target)
    initial_path  = os.getcwd()
    os.chdir('Data')
    df = pd.read_csv("{} {}.csv".format(list(request_body[0].values())[1],
    list(request_body[0].values())[0]), index_col=0)
    
    print('========================')
    print(df)
    columns = df.columns
    print(columns)
    print(list(columns))
    scaler = MinMaxScaler()
    X = scaler.fit(df).transform(df)
    #scaler.fit(df)
    data_set = pd.DataFrame(X, columns=columns)
    print('DATASET IS:::')
    print(data_set)
   
    
    
    os.chdir(initial_path)
    with open('scalers/' + warehouse + "_" + target+ '_scaler.pkl', 'wb') as fid:
            pickle.dump(scaler, fid)

    print('scaler for ', warehouse, target,' exists')
    init_data = df.copy()
    #load the scaler
    #scaler = load(open('scalers/' + self.protocol + "_" + self.model_type +'_scaler.pkl', 'rb'))
    #X = scaler.fit(X).transform(X)
    #all_set = pd.DataFrame(scaler.transform(all_set), columns=columns)
   
    #os.chdir(initial_path)
    train_task = Process(target=train, args=(warehouse, target, init_data, data_set, n_filters, filter_width, batch_size, windowsize,
    activation_function, timestep, numepochs))
   
   
    train_task.start()
    #train(warehouse, target, data_set, n_filters, filter_width, batch_size, windowsize, activation_function, timestep, numepochs)
    return 'Training process is started', 200

@app.route("/Inference", methods=['POST'])
def start_inference():
    request_body = request.get_json()
    print(request_body)
    print(list(request_body[0].values()))
    print(type(request_body[0]))
   
    sections = config.keys()
    warehouse = list(request_body[0].values())[1]
    target = list(request_body[0].values())[0]
    order = list(request_body[0].values())[2]
    window_size = int(config[target][warehouse ]['windows_size'])
    print(type(window_size))
    #window_size: object = windowsize
    time_step = int(config[target][warehouse ]['timestep'])
    inference_process = Process(target=inference, args=(warehouse, target, order, window_size, time_step))
    inference_process .start()
    return 'Prediction is estimated', 200

def inference(warehouse, target, order, window_size, time_step):
    initial_path = os.getcwd()
    os.chdir('Data')
    df = pd.read_csv("{} {}.csv".format(warehouse,
    target), index_col=0)
    os.chdir(initial_path)
    init_data = df.copy()
    model3 = tf.keras.models.load_model('{}/models/{} {}.pkl'.format(os.getcwd(), warehouse, target))
    model3.summary()
    #order = '25/04/2018 11:44:59'


    date_time_obj = datetime.strptime(order, '%d/%m/%Y %H:%M:%S')

    weekday = np.int64(date_time_obj.weekday())
    day = np.int64(date_time_obj.day)
    month = np.int64(date_time_obj.month)
    print(weekday)
    print(type(weekday))
    print(day)
    print(type(day))
    print(month)
    print(type(month))

    columns = list(df.columns)#[:-1]
    print(columns)

    print(list(columns))
    print(type(columns))
    values = [0] * len(columns)
    values[0] = month
    values[1] = day
    values[2] = weekday
    #values[2] = day
    #values[4] = weekday
    print(values)
    pos = columns.index('{}'.format(warehouse))
    print(pos)
    values[pos] = 1
    print(values)

    df1 = pd.DataFrame ([values], columns = columns)
    print(df1)
    print(df)
    df = pd.concat([init_data[columns], df1]).reset_index(drop=True)
    print(df)
    print(df['Ekol Polonya'])
    os.chdir(initial_path)
    scaler = load(open('scalers/' + warehouse + "_" + target+ '_scaler.pkl', 'rb'))
    X = scaler.fit(df).transform(df)
    print(X)
    print(X[-1])
    print(len(X))
    num_features = len(df.columns) - 1
    num_df = df.to_numpy()
    res_x, res_y = [], []

    for i in range(len(df) - window_size - 1):
        res_x.append(num_df[i:i + window_size, :num_features])

        res_y.append(np.array(num_df[(i + window_size + 1):(i + window_size + time_step + 1), num_features]))
    res_x = res_x[:len(res_x) - time_step]
    res_x2 = np.asarray(res_x)
    res_x2 = res_x2.reshape(res_x2.shape[0], res_x2.shape[1], num_features)
    preds = model3.predict(res_x2)
    print(preds)
    prediction = abs(preds[-1][0][0])
    print(prediction)
    min = df.iloc[:,-1].min()
    max =  df.iloc[:,-1].max()
    print(min)
    print(max)
    final_prediction = (prediction * max + (1 - prediction) * min)
    print(final_prediction)
    estimated_departure  = date_time_obj + timedelta(hours=int(final_prediction))
    print(estimated_departure)
    estimated_departure_str = estimated_departure.strftime('%d/%m/%Y %H:%M:%S')
    print('Final Time as string object: ', estimated_departure_str)
    return estimated_departure_str

#result_file = config['data_paths']['result_file']
def rmse(y_true, y_pred):
   return backend.sqrt(backend.mean((backend.square(y_pred - y_true)), axis=-1))#/statistics.stdev(y_true)
#r'''
def train(warehouse, target, init_data, data_set, n_filters, filter_width, batch_size, windowsize, activation_function, timestep, numepochs):
    df = data_set.copy().reset_index(drop=True)
    print('dataset is:')
    print(df)
    initial_path = os.getcwd()
    #df = pd.read_csv(data_set, index_col=0)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    df = df.dropna().reset_index(drop=True)
    window_size = windowsize
    print(type(window_size))
    #window_size: object = windowsize
    time_step = timestep
    num_features = len(df.columns) -1
    num_df = df.to_numpy()
    res_x, res_y = [], []
    for i in range(len(df) - window_size - 1):
        res_x.append(num_df[i:i + window_size, : len(df.columns)-1 ])

        res_y.append(np.array(num_df[(i + window_size + 1):(i + window_size + time_step + 1), len(df.columns) -1]))

    res_x = res_x[:len(res_x) - time_step]
    res_y = res_y[:len(res_y) - time_step]
    res_x2 = np.asarray(res_x)#.astype('float32')
    res_x2 = res_x2.reshape(res_x2.shape[0], res_x2.shape[1], num_features)
    res_y2 = np.vstack(res_y)
    res_y2 = res_y2.reshape(res_y2.shape[0], res_y2.shape[1], 1)
    train_x, test_x, train_y, test_y = train_test_split(res_x2, res_y2, test_size=0.3)
    print(res_x2)
    print (type(res_y2))
    n_filters = n_filters
    filter_width = filter_width
    dilation_rates = [2 ** i for i in range(8)]

    history_seq = Input(shape=(windowsize, len(df.columns) -1))#57
    x = history_seq
    skips = []
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                   kernel_size=filter_width,
                   padding='causal',
                   dilation_rate=dilation_rate)(x)
    x = Dense(40, activation= activation_function)(x)
    x = Dense(20, activation= activation_function)(x)
    x = Dropout(.1)(x)
    x = Dense(1)(x)

    def slice(x, seq_length):
        return x[:, -seq_length:, :]

    pred_seq_train = Lambda(slice, arguments={'seq_length': time_step})(x)

    model = Model(history_seq, pred_seq_train)
    batch_size = batch_size#64
    epochs = numepochs

    model.compile(Adam(), loss='mean_squared_error' )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    history = model.fit(res_x2, res_y2,
                        batch_size=batch_size,
                        epochs=epochs, validation_data=(test_x, test_y), callbacks=[es])
    #history = model.fit(train_x, train_y,
                        #batch_size=batch_size,
                        #epochs=epochs, validation_data=(test_x, test_y), callbacks=[es])

    for i in history.history:
        print(i, history.history[i])


    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    mseTestSetHidden = model.evaluate(test_x, test_y)  # [1]
    rmseTestSetHidden = np.sqrt(mseTestSetHidden)
    nrmseTestSetHiddenSD = rmseTestSetHidden / np.std(test_y)
    print('mse = ', str(mseTestSetHidden))
    print('rmse = ', str(rmseTestSetHidden))
    print('nrmseSD = ', str(nrmseTestSetHiddenSD))
    # print('nrmseMean = ', str(nrmseTestSetHiddenMean))
    # print('nrmseMaxMin = ', str(nrmseTestSetHiddenMaxMin))
    print(model.evaluate(test_x, test_y))

    #f = open(os.getcwd() + '.pkl', "wb")
    f = open('{}/history/{} {}.pkl'.format(os.getcwd(), warehouse, target), "wb")
    pickle.dump(history.history, f)
    print(f.name)

    f.close()
    model.save('{}/models/{} {}.pkl'.format(os.getcwd(), warehouse, target))
    


    r'''
    model3 = tf.keras.models.load_model('{}/models/{} {}.pkl'.format(os.getcwd(), warehouse, target))
    model3.summary()
    order = '25/04/2018 11:44:59'


    date_time_obj = datetime.strptime(order, '%d/%m/%Y %H:%M:%S')

    #print(date_time_obj. month)
    #print(type(date_time_obj. month))

    weekday = np.int64(date_time_obj.weekday())
    day = np.int64(date_time_obj.day)
    month = np.int64(date_time_obj.month)
    print(weekday)
    print(type(weekday))
    print(day)
    print(type(day))
    print(month)
    print(type(month))

    columns = list(df.columns)#[:-1]
    print(columns)

    print(list(columns))
    print(type(columns))
    values = [0] * len(columns)
    values[0] = month
    values[1] = day
    values[2] = weekday
    #values[2] = day
    #values[4] = weekday
    print(values)
    pos = columns.index('{}'.format(warehouse))
    print(pos)
    values[pos] = 1
    print(values)

    df1 = pd.DataFrame ([values], columns = columns)
    print(df1)
    print(df)
    df = pd.concat([init_data[columns], df1]).reset_index(drop=True)
    print(df)
    print(df['Ekol Polonya'])
    os.chdir(initial_path)
    #warehouse = 'Ekol Polonya'
    #target = 'DEPARTURE-ORDER'
    scaler = load(open('scalers/' + warehouse + "_" + target+ '_scaler.pkl', 'rb'))
    X = scaler.fit(df).transform(df)
    print(X)
    print(X[-1])
    print(len(X))
    num_features = len(df.columns) - 1
    num_df = df.to_numpy()
    res_x, res_y = [], []

    for i in range(len(df) - window_size - 1):
        res_x.append(num_df[i:i + window_size, :num_features])

        res_y.append(np.array(num_df[(i + window_size + 1):(i + window_size + time_step + 1), num_features]))
    res_x = res_x[:len(res_x) - time_step]
    res_x2 = np.asarray(res_x)
    res_x2 = res_x2.reshape(res_x2.shape[0], res_x2.shape[1], num_features)
    preds = model3.predict(res_x2)
    print(preds)
    prediction = abs(preds[-1][0][0])
    print(prediction)
    min = df.iloc[:,-1].min()
    max =  df.iloc[:,-1].max()
    print(min)
    print(max)
    final_prediction = (prediction * max + (1 - prediction) * min)
    print(final_prediction)
    estimated_departure  = date_time_obj + timedelta(hours=int(final_prediction))
    print(estimated_departure)
    estimated_departure_str = estimated_departure.strftime('%d/%m/%Y %H:%M:%S')
    print('Final Time as string object: ', estimated_departure_str)
    '''
app.run(host='localhost', port=5003)
#cash_list = ['US 10Treasure']

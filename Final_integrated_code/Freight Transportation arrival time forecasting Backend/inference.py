import mysql.connector as mySQL
from datetime import datetime, timezone, timedelta
from numpy import array
from numpy import hstack
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
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
from flask_cors import CORS
from configobj import ConfigObj
from flask import Flask, request, jsonify, json, render_template,redirect, url_for
import random
import time
#import jwt
config_file = 'config.ini'
config = ConfigObj(config_file,list_values=True,interpolation=True)
sections = config.keys()

app = Flask(__name__)
CORS(app)
@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello from big_smart_log app!!</h1>"

@app.route("/login", methods=['POST'])
def login():
    
    try:
        request_body = request.get_json()
        mydb = getDatabaseConnection()
        cursor = mydb.cursor()
        select_query = "SELECT * FROM users"
        ##select_query = "SELECT * FROM {}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), target.split('-')[0].lower(), target.split('-')[1].lower())
        cursor.execute(select_query)
        all_assets = cursor.fetchall()
        result = pd.DataFrame(data = all_assets,columns = ['email','passwd'] )
        filter1 = result["email"]== list(request_body[0].values())[0]
        # making boolean series for age
        filter2 = result["passwd"]== list(request_body[0].values())[1]
        #filtering data on basis of both filters
        result.where(filter1 & filter2, inplace = True)
        result = result.dropna()
        print('XXXXXXXXXXXXXXXXXXXXXXXX')
        print(result)
        print(type(result))
        #print(type(result.iloc[0]['email']))
        print(os.getcwd())
        # display
        #if result.iloc[0]['email'] is np.nan or result.iloc[0]['passwd'] is np.nan:
        if result.empty:
            print('The dataframe is empty')
            return jsonify({}) 
            #return  jsonify({"401 Error: There was a problem with your request. Your e-mail or your password is not validated. Please correct and try again.":401})
        else:
            r'''
            print(result)
            my_key = 'my_key'
            result = result.to_json(orient="records")
            parsed = json.loads(result)
            print(parsed)
            token = jwt.encode(
                payload=parsed[0],
                key=my_key
            )
            print(token)
            '''
            #response = json.dumps({"The user is valid: 200"}), 200
            #return jsonify({"The user is valid": 200})
            return jsonify({"token":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.Et9HFtf9R3GEMA0IICOfFMVXY7kkTX1wr4qCyhIf58U"})
            ##return jsonify({"token":token})
    except Exception as e:
        print("====================" + str(e) + "====================")
        return e

@app.route("/RequestResult", methods=['POST'])
def post_selected_result():
    mydb = getDatabaseConnection()
    cursor = mydb.cursor()
    request_body = request.get_json()
    #r'''
    stmt = "SHOW TABLES LIKE 'appeared_result'"
    cursor.execute(stmt)
    existment = cursor.fetchone()
    if existment:
        query = "DROP TABLE appeared_result"
        cursor.execute(query)
        cursor.execute("CREATE TABLE appeared_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255),WH5 VARCHAR(255))")
    else:
        cursor.execute("CREATE TABLE appeared_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255),WH5 VARCHAR(255))")
        #sql = "INSERT INTO ShipmentPrediction (id,date,arrayNum) " 
        #sql += " VALUES (%s, %s, %s )"
        #values = ('Shipment 1', 'DD/MM/YYYY HH:MM:SS', 1)
        #cursor.execute(sql, values)
        #mydb.commit()
    #'''
    pick_stmt = "SHOW TABLES LIKE 'pick_appeared_result'"
    cursor.execute(pick_stmt)
    pick_existment = cursor.fetchone()
    if pick_existment:
        query = "DROP TABLE pick_appeared_result"
        cursor.execute(query)
        cursor.execute("CREATE TABLE pick_appeared_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255),WH5 VARCHAR(255))")
    else:
        cursor.execute("CREATE TABLE pick_appeared_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255),WH5 VARCHAR(255))")
    
    eta_stmt = "SHOW TABLES LIKE 'eta_appeared_result'"
    cursor.execute(eta_stmt)
    eta_existment = cursor.fetchone()
    if eta_existment:
        query = "DROP TABLE eta_appeared_result"
        cursor.execute(query)
        cursor.execute("CREATE TABLE eta_appeared_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255),WH5 VARCHAR(255))")
    else:
        cursor.execute("CREATE TABLE eta_appeared_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255),WH5 VARCHAR(255))")
    #query = "SELECT * FROM result WHERE shipment = %s AND date=%s"
    query = "SELECT DISTINCT * FROM result WHERE shipment = %s AND date=%s"#"DROP TABLE result"
    print(list(request_body[0].values())[0])
    values = (list(request_body[0].values())[0],list(request_body[0].values())[1])
    cursor.execute(query, values)
    all_assets = cursor.fetchall()
    result = pd.DataFrame(data = all_assets,columns = ['shipment','date','id','WH1','WH2','WH3','WH4','WH5'] )#.iloc[:,1:]
    for i in range (0, len(result)):
        query = "INSERT INTO appeared_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5)"
        query += " VALUES (%s,%s, %s, %s,%s, %s, %s, %s)"
        values = (result.iloc[i]['shipment'],result.iloc[i]['date'], result.iloc[i]['id'],result.iloc[i]['WH1'],result.iloc[i]['WH2'],result.iloc[i]['WH3'],result.iloc[i]['WH4'],result.iloc[i]['WH5'])
        print(values)
        cursor.execute(query, values)
        mydb.commit()
    #pquery = "SELECT * FROM pick_result WHERE shipment = %s AND date=%s"
    pquery = "SELECT DISTINCT * FROM pick_result WHERE shipment = %s AND date=%s"
    print(list(request_body[0].values())[0],list(request_body[0].values())[1])
    values = (list(request_body[0].values())[0],list(request_body[0].values())[1])
    cursor.execute(pquery, values)
    all_picks = cursor.fetchall()
    presult = pd.DataFrame(data = all_picks,columns = ['shipment','date','id','WH1','WH2','WH3','WH4','WH5'] )#.iloc[:,1:]
    for i in range (0, len(presult)):
        p_query = "INSERT INTO pick_appeared_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5)"
        p_query += " VALUES (%s, %s,%s, %s,%s, %s, %s, %s)"
        values = (presult.iloc[i]['shipment'], presult.iloc[i]['date'], presult.iloc[i]['id'],presult.iloc[i]['WH1'],presult.iloc[i]['WH2'],presult.iloc[i]['WH3'],presult.iloc[i]['WH4'],presult.iloc[i]['WH5'])
        print(values)
        cursor.execute(p_query, values)
        mydb.commit()

    equery = "SELECT DISTINCT * FROM eta_result WHERE shipment = %s AND date=%s"
    print(list(request_body[0].values())[0],list(request_body[0].values())[1])
    values = (list(request_body[0].values())[0],list(request_body[0].values())[1])
    cursor.execute(equery, values)
    all_eta = cursor.fetchall()
    eresult = pd.DataFrame(data = all_eta,columns = ['shipment','date','id','WH1','WH2','WH3','WH4','WH5'] )#.iloc[:,1:]
    for i in range (0, len(eresult)):
        e_query = "INSERT INTO eta_appeared_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5)"
        e_query += " VALUES (%s, %s,%s, %s,%s, %s, %s, %s)"
        values = (eresult.iloc[i]['shipment'], eresult.iloc[i]['date'], eresult.iloc[i]['id'],eresult.iloc[i]['WH1'],eresult.iloc[i]['WH2'],eresult.iloc[i]['WH3'],eresult.iloc[i]['WH4'],eresult.iloc[i]['WH5'])
        print(values)
        cursor.execute(e_query, values)
        mydb.commit()
    return jsonify({"OK":200})

@app.route("/GetRequestedResult", methods=['GET'])
def get_selected_result():
    mydb = getDatabaseConnection()
    cursor = mydb.cursor()

    nrmse_query = "SELECT * FROM nrmse_delivery"
    cursor.execute(nrmse_query)
    all_assets = cursor.fetchall()
    nrmse_data = pd.DataFrame(data = all_assets,columns = config['columns']['nrmse'])#.iloc[:,1:]
    print(nrmse_data)
    nrmse_result = nrmse_data.to_json(orient="records")
    nrmse_parsed = json.loads(nrmse_result)
    print(nrmse_parsed )
    print(type(nrmse_parsed [0]))
    #final_table = jsonify(parsed) + jsonify(nrmse_parsed)
    #print(final_table)
    #return jsonify(pick, parsed, nrmse_parsed)

    nrmse_dep_query = "SELECT * FROM nrmse_departure"
    cursor.execute(nrmse_dep_query)
    all_assets = cursor.fetchall()
    nrmse_dep_data = pd.DataFrame(data = all_assets,columns = config['columns']['nrmse'])#.iloc[:,1:]
    print(nrmse_dep_data)
    nrmse_dep_result = nrmse_dep_data.to_json(orient="records")
    nrmse_dep_parsed = json.loads(nrmse_dep_result)
    print(nrmse_dep_parsed)
    print(type(nrmse_dep_parsed [0]))
    #final_table = jsonify(parsed) + jsonify(nrmse_parsed)
    #print(final_table)
    #return jsonify(pick, parsed, nrmse_parsed)

    nrmse_eta_query = "SELECT * FROM nrmse_eta_delivery"
    cursor.execute(nrmse_eta_query)
    all_assets = cursor.fetchall()
    nrmse_eta_data = pd.DataFrame(data = all_assets,columns = config['columns']['nrmse'])#.iloc[:,1:]
    print(nrmse_eta_data)
    nrmse_eta_result = nrmse_eta_data.to_json(orient="records")
    nrmse_eta_parsed = json.loads(nrmse_eta_result)
    print(nrmse_eta_parsed)
    print(type(nrmse_eta_parsed [0]))


    stmt = "SHOW TABLES LIKE 'appeared_result'"
    cursor.execute(stmt)
    existment = cursor.fetchone()

    pick_stmt = "SHOW TABLES LIKE 'pick_appeared_result'"
    cursor.execute(pick_stmt)
    pick_existment = cursor.fetchone()

    eta_stmt = "SHOW TABLES LIKE 'eta_appeared_result'"
    cursor.execute(eta_stmt)
    eta_existment = cursor.fetchone()
    
    if existment:
        dataframe = pd.read_sql("""SELECT * FROM appeared_result""", con = mydb).iloc[:,1:]
        print('==================================================')
        print(dataframe)
        result = dataframe.to_json(orient="records")
        parsed = json.loads(result)
    else:  
        parsed = {}
    if pick_stmt:
        pick_dataframe = pd.read_sql("""SELECT * FROM pick_appeared_result""", con = mydb).iloc[:,1:]
        print('==================================================')
        print(pick_dataframe)
        pickresult = pick_dataframe.to_json(orient="records")
        pick = json.loads(pickresult)
        #parsed = json.dumps(parsed, indent=4) 
        #return jsonify(parsed)
    else:  
        pick = {}
        #return jsonify({}) 
    if eta_existment:
        eta_dataframe = pd.read_sql("""SELECT * FROM eta_appeared_result""", con = mydb).iloc[:,1:]
        print('==================================================')
        print(eta_dataframe)
        etaresult = eta_dataframe.to_json(orient="records")
        eta = json.loads(etaresult)
        #parsed = json.dumps(parsed, indent=4) 
        #return jsonify(parsed)
    else:  
        eta = {}
        #return jsonify({}) 
    return jsonify(pick,nrmse_dep_parsed,eta,nrmse_eta_parsed,parsed,nrmse_parsed)
@app.route("/EditItemDatabase", methods=['POST'])
def EditDataPoint():
    try:
        request_body = request.get_json()
        print(request_body)
        print(list(request_body[0].values())[0])
        print(type(list(request_body[0].values())[0]))
        print(list(request_body[0].values())[1])
        print(type(list(request_body[0].values())[1]))
        print("0")
        shipmentID = list(request_body[0].values())[0]
        PredictionRunned = list(request_body[0].values())[1]
        mydb = getDatabaseConnection()
        cursor = mydb.cursor()
        stmt = "SHOW TABLES LIKE 'ShipmentPrediction'"
        cursor.execute(stmt)
        existment = cursor.fetchone()
        if existment:
            pass
        else:
            cursor.execute("CREATE TABLE ShipmentPrediction (id VARCHAR(255), date VARCHAR(255), arrayNum INT)")
        #"UPDATE bigsmartlog.shipmentprediction SET ShipmentID = "
        query = "UPDATE shipmentprediction SET id = %s WHERE date = %s"
        values = (list(request_body[0].values())[0], list(request_body[0].values())[1])
        print(query)
        print(values)
        cursor.execute(query, values)
        mydb.commit()
        res_query = "UPDATE result SET shipment = %s WHERE date = %s"
        res_values = (list(request_body[0].values())[0], list(request_body[0].values())[1])
        print(res_query)
        print(res_values)
        cursor.execute(res_query, res_values)
        mydb.commit()

        pick_query = "UPDATE pick_result SET shipment = %s WHERE date = %s"
        pick_values = (list(request_body[0].values())[0], list(request_body[0].values())[1])
        print(pick_query)
        print(pick_values)
        cursor.execute(pick_query, pick_values)
        mydb.commit()

        eta_query = "UPDATE eta_result SET shipment = %s WHERE date = %s"
        eta_values = (list(request_body[0].values())[0], list(request_body[0].values())[1])
        print(eta_query)
        print(eta_values)
        cursor.execute(eta_query, eta_values)
        mydb.commit()
    except Exception as e:
        print("====================" + str(e) + "====================")
    finally:
        cursor.close()
        mydb.close()
            
    return jsonify({"The record is modified":200})
    


@app.route("/AddItemDatabase", methods=['POST'])

def add_new_item():
        try:
            #frame   = dataFrame.to_sql(tableName, dbConnection, if_exists='fail');
            request_body = request.get_json()
            
            #result = []
            mydb = getDatabaseConnection()
            cursor = mydb.cursor()
            stmt = "SHOW TABLES LIKE 'ShipmentPrediction'"
            cursor.execute(stmt)
            existment = cursor.fetchone()
            if existment:
                pass
            else:
                cursor.execute("CREATE TABLE ShipmentPrediction (id VARCHAR(255), date VARCHAR(255), arrayNum INT)")

            #result = "{}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), target.split('-')[0].lower(), target.split('-')[1].lower())
            select_query = "SELECT * FROM ShipmentPrediction"
            ##select_query = "SELECT * FROM {}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), target.split('-')[0].lower(), target.split('-')[1].lower())
            cursor.execute(select_query)
            all_assets = cursor.fetchall()
            result = pd.DataFrame(data = all_assets,columns = ['id','date','arrayNum'] )#.iloc[:,1:]
            
            print(result)
            if result.empty == True:
                mydb = getDatabaseConnection()
                cursor = mydb.cursor()
                print('DataFrame is empty')
                sql = "INSERT INTO ShipmentPrediction (id,date,arrayNum) " 
                sql += " VALUES (%s, %s, %s )"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'), 1)
                values = ('{}'.format(list(request_body[0].values())[0]), datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 1) #now.strftime("%m/%d/%Y, %H:%M:%S")
                print(sql)
                cursor.execute(sql, values)
                mydb.commit()
                success = True
                all_assets = cursor.fetchall()
                result = pd.DataFrame(data = all_assets,columns = ['id','date','arrayNum'] )#.iloc[:,1:]
                #new_dict = {'ShipmentID': 'Shipment 1', 'PredictionRunned': datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S')}
                #result = result.append(new_dict, ignore_index = True)
                print(result)
                #result.to_sql('ShipmentPrediction', mydb, index=False)
                
            else:
                success = False
                print('DataFrame is not empty')
                select_query = "SELECT * FROM ShipmentPrediction"
                cursor.execute(select_query)
                all_assets = cursor.fetchall()
                result = pd.DataFrame(data = all_assets,columns = ['id','date','arrayNum'] )#.iloc[:,1:]
                print(result)
                #print(int(result.iloc[-1]['id'].split()[-1]))

                sql = "INSERT INTO ShipmentPrediction (id,date,arrayNum) " 
                sql += " VALUES (%s, %s, %s)"
                #values = ('Shipment {}'.format(str(int(result.iloc[-1]['id'].split()[-1])+1)), datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'),int(result.iloc[-1]['id'].split()[-1])+1)
                values = ('{}'.format(list(request_body[0].values())[0]), datetime.now().strftime("%d/%m/%Y %H:%M:%S") ,int(result.iloc[-1]['arrayNum'])+1)
                print(sql)
                print(values)
                cursor.execute(sql, values)
                mydb.commit()
                success = True
                all_assets = cursor.fetchall()
                result = pd.DataFrame(data = all_assets,columns = ['id','date','arrayNum'] )#
                print(result)

                
        
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
        
        #return 'The database is activate',200#result#
        return jsonify({"The record is added":200})


@app.route("/DatabaseDelete", methods=['POST'])
def delete_assets():
        try:
            #frame   = dataFrame.to_sql(tableName, dbConnection, if_exists='fail');
            request_body = request.get_json()
            mydb = getDatabaseConnection()
            cursor = mydb.cursor()
            del_date = list(request_body[0].values())[0]
            #print(del_id)
            # Delete a record
            #sql_Delete_query = """Delete from ShipmentPrediction where id = 'Shipment {}' """.format(del_id)
            
            sql_Delete_query = """Delete from ShipmentPrediction where date = '{}' """.format(del_date)
            cursor.execute(sql_Delete_query)#, deleted_row
            mydb.commit()
            stmt = "SHOW TABLES LIKE 'result'"
            cursor.execute(stmt)
            existment = cursor.fetchone()
            if existment:
               print("There is the table")
               result_delete_query = """Delete from result where date = '{}' """.format(del_date)
               cursor.execute(result_delete_query )#, deleted_row
               mydb.commit()
            #else:

            pick_stmt = "SHOW TABLES LIKE 'pick_result'"
            cursor.execute(pick_stmt)
            pick_existment = cursor.fetchone()
            if pick_existment:
               print("There is the table")
               pick_delete_query = """Delete from pick_result where date = '{}' """.format(del_date)
               cursor.execute(pick_delete_query)#, deleted_row
               mydb.commit()
            eta_stmt = "SHOW TABLES LIKE 'eta_result'"
            cursor.execute(eta_stmt)
            eta_existment = cursor.fetchone()
            if eta_existment:
               print("There is the table")
               eta_delete_query = """Delete from eta_result where date = '{}' """.format(del_date)
               cursor.execute(eta_delete_query)#, deleted_row
               mydb.commit()
            #else:
            r'''    
            select_query = "SELECT * FROM ShipmentPrediction"
            cursor.execute(select_query)
            all_assets = cursor.fetchall()
            result = pd.DataFrame(data = all_assets,columns = ['id','date','arrayNum'] )#.iloc[:,1:]
            print('============================')
            print(result)
            cnt = 0
            res_cnt = 1
            for i in range (0, len(result)):
                print(result.iloc[i]['id'])
                temp_res = result.copy()
                #result.iloc[i]['id'] = 'Shipment {}' .format(str(cnt+1))
                result.at[i,'id'] = 'Shipment {}' .format(str(cnt+1))
                result.at[i,'arrayNum'] = (cnt+1)
                print(result.iloc[i]['id'])
                print(result.iloc[i]['arrayNum'])
                print(type(result.iloc[i]['arrayNum']))
                query = "UPDATE ShipmentPrediction SET id = %s, arrayNum = %s WHERE id = %s"#date = %s
                #query+= "SET ShipmentID= '%s''"
                #query+=  "WHERE PredictionRunned = '%s'"#.format(result.iloc[i]['PredictionRunned'])
                #values = (result.iloc[i]['id'], np.int64(result.iloc[i]['arrayNum']).item(),result.iloc[i]['date'])
                values = (result.iloc[i]['id'], np.int64(result.iloc[i]['arrayNum']).item(),temp_res.iloc[i]['id'])
                print(values)
                cursor.execute(query, values)
                mydb.commit()
                if existment:
                    print("There is the table")
                
                    result_query = "UPDATE result SET shipment = %s WHERE shipment = %s"  
                    #query+= "SET ShipmentID= '%s''"
                    #query+=  "WHERE PredictionRunned = '%s'"#.format(result.iloc[i]['PredictionRunned'])
                    values = (result.iloc[i]['id'], temp_res.iloc[i]['id'])
                    cursor.execute(result_query, values)
                    mydb.commit()
                    
                    cnt+=1
                    res_cnt+=1
            
            print(result)              
            '''    
        except Exception as e:
            print("====================" + str(e) + "====================")
        finally:
            cursor.close()
            mydb.close()
        
        #return 'The database is activate',200#result#
        return jsonify({"The record is deleted":"200"})
@app.route("/DatabaseAppear", methods=['GET'])
def get_db():
    mydb = getDatabaseConnection()
    cursor = mydb.cursor()
    stmt = "SHOW TABLES LIKE 'ShipmentPrediction'"
    cursor.execute(stmt)
    existment = cursor.fetchone()
    if existment:
        pass
    else:
        cursor.execute("CREATE TABLE ShipmentPrediction (id VARCHAR(255), date VARCHAR(255), arrayNum INT)")
        sql = "INSERT INTO ShipmentPrediction (id,date,arrayNum) " 
        sql += " VALUES (%s, %s, %s )"
        values = ('Shipment 1',datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'), 1)
        cursor.execute(sql, values)
        mydb.commit()
    #dataframe = pd.read_sql("""SELECT id,date FROM ShipmentPrediction""", con = mydb)
    dataframe = pd.read_sql("""SELECT * FROM ShipmentPrediction""", con = mydb)
    print('==================================================')
    print(dataframe)
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    #parsed = json.dumps(parsed, indent=4) 
    return jsonify(parsed)

#In this project the purpose is to create a response like [{'id': 'RF', 'WH1': '22/03/2022 13:48:10', 'WH2': '21/03/2022 11:48:10', 'WH3': '22/03/2022 17:48:10', 'WH4': '22/03/2022 07:48:10', 'WH5': '23/03/2022 13:48:10'},
# {'id': 'WN', 'WH1': '25/03/2022 11:48:10', 'WH2': '20/03/2022 23:48:10', 'WH3': '10/07/2022 20:48:10', 'WH4': '24/03/2022 14:48:10', 'WH5': '01/04/2022 05:48:10'}]
@app.route("/Inferencedelivery", methods=['POST'])
def start_del_est():
    sections = config.keys()
    #We define the target the warehouses which we are going to resd from config file
    target = 'DEL_TRUE-DEPARTURE'
    warehouses = config['warehouses']['wh']
    print(warehouses)
    print(type(warehouses))
    #We define the range of models by config file
    models = config['models']['models']#.split(',')
    print(models)
    print(type(models))
    #The order is when the user click the button means that this is the current timestamp
    #order = datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S') 
    request_body = request.get_json()
    shipment = list(request_body[0].values())[0]
    order = list(request_body[0].values())[1]
    
    #In this list it is going to load the results
    json_result = []
    pick_result = []
    json_eta = []
    for model in models:
        #Initial json object to get the results of each model
        med_json = {}
        med_json_pick = {}
        med_json_eta = {}
        #The id feature of each json is the name of the model
        med_json["id"] = model
        med_json["shipment"] = shipment
        med_json_pick["id"] = model
        med_json_pick["shipment"] = shipment
        med_json_eta["id"] = model
        med_json_eta["shipment"] = shipment
        for  warehouse in warehouses:
            print(warehouse)
            #warehouse is the hidden names like WH1, WH2 etc real_warehouse is the names of warehouses like EKol Czech, Ekol Hungary
            if warehouse == 'WH1':
                real_warehouse = 'Ekol Czech'
            elif warehouse == 'WH2':
                real_warehouse = 'Ekol Hungary'
            elif warehouse == 'WH3':
                real_warehouse = 'Ekol Polonya'
            elif warehouse == 'WH4':
                real_warehouse = 'Ostrava Railport'
            elif warehouse == 'WH5':
                real_warehouse = 'Trieste Railport'
            
            if model == 'RF':
                #Get the list of the new data point based on DEPARTURE-ORDER dataframe and the order datatime(date_time_obj). The order datetime is the datetime.now()
               dep_values, date_time_obj = departure_dates(order, 'DEPARTURE-ORDER',  real_warehouse, model)
               #we predict the departure datetime (DEPARTURE-ORDER case) by the Random Forest model
               departure = RandomForest(date_time_obj, real_warehouse, dep_values, 'departure')#inference_departure(real_warehouse, 'DEPARTURE-ORDER', model, order, dep_values, date_time_obj)
               med_json_pick["{}".format(warehouse)] = departure
               #Get the list of the new data point based on ETA_DELIVERY-DEPARTURE dataframea nd the departure datetime(date_time_obj). 
               values, date_time_departure = delivery_dates(order, departure, target, real_warehouse, model)
               med_json["{}".format(warehouse)] = RandomForest(date_time_departure, real_warehouse, values, 'delivery')
               med_json_eta["{}".format(warehouse)]= (datetime.utcfromtimestamp(time.mktime(datetime.strptime(med_json["{}".format(warehouse)], '%d/%m/%Y %H:%M:%S').timetuple()) + random.randrange(-129600, 129600, 1)).strftime('%d/%m/%Y %H:%M:%S'))#.strftime('%d/%m/%Y %H:%M:%S')
            #The same process as above made for Wavenet model
            if model == 'WN':
                initial_path = os.getcwd()
                window_size = int(config[target][real_warehouse ]['windows_size'])
                time_step = int(config[target][real_warehouse ]['timestep'])
                dep_values, date_time_obj = departure_dates(order, 'DEPARTURE-ORDER',  real_warehouse, model)
                departure = wavenet(date_time_obj, real_warehouse,'DEPARTURE-ORDER',dep_values)#inference_departure(real_warehouse, 'DEPARTURE-ORDER', model, order)
                med_json_pick["{}".format(warehouse)] = departure
                values, date_time_departure =  delivery_dates(order, departure, target, real_warehouse, model)
                med_json["{}".format(warehouse)] = wavenet(date_time_departure, real_warehouse,target, values)
                med_json_eta["{}".format(warehouse)]= (datetime.utcfromtimestamp(time.mktime(datetime.strptime(med_json["{}".format(warehouse)], '%d/%m/%Y %H:%M:%S').timetuple()) + random.randrange(-129600, 129600, 1)).strftime('%d/%m/%Y %H:%M:%S'))#.strftime('%d/%m/%Y %H:%M:%S')
            if model == 'B':
               print('Model is B')
               #Get the list of the new data point based on DEPARTURE-ORDER dataframe and the order datatime(date_time_obj). The order datetime is the datetime.now()
               dep_values, date_time_obj = departure_dates(order, 'DEPARTURE-ORDER',  real_warehouse, model)
               #we predict the departure datetime (DEPARTURE-ORDER case) by the Random Forest model
               #departure = bagging(date_time_obj, real_warehouse,'departure', dep_values)
               departure = bagging(date_time_obj, real_warehouse,'departure',dep_values,'DEPARTURE-ORDER')#inference_departure(real_warehouse, 'DEPARTURE-ORDER', model, order, dep_values, date_time_obj)
               med_json_pick["{}".format(warehouse)] = departure
               print(departure)
               values, date_time_departure = delivery_dates(order, departure, target, real_warehouse, model)
               med_json["{}".format(warehouse)] = bagging(date_time_departure, real_warehouse, 'delivery',values, 'DEL_TRUE-DEPARTURE')
               med_json_eta["{}".format(warehouse)]= (datetime.utcfromtimestamp(time.mktime(datetime.strptime(med_json["{}".format(warehouse)], '%d/%m/%Y %H:%M:%S').timetuple()) + random.randrange(-129600, 129600, 1)).strftime('%d/%m/%Y %H:%M:%S'))#.strftime('%d/%m/%Y %H:%M:%S')
            #The same process as above made for Wavenet model
            if model == 'GB':
               print('Model is GB')
               #Get the list of the new data point based on DEPARTURE-ORDER dataframe and the order datatime(date_time_obj). The order datetime is the datetime.now()
               dep_values, date_time_obj = departure_dates(order, 'DEPARTURE-ORDER',  real_warehouse, model)
               #we predict the departure datetime (DEPARTURE-ORDER case) by the Random Forest model
               #departure = bagging(date_time_obj, real_warehouse,'departure', dep_values)
               departure = GradientBoosting(date_time_obj, real_warehouse,'departure',dep_values,'DEPARTURE-ORDER')#inference_departure(real_warehouse, 'DEPARTURE-ORDER', model, order, dep_values, date_time_obj)
               med_json_pick["{}".format(warehouse)] = departure
               print(departure)
               values, date_time_departure = delivery_dates(order, departure, target, real_warehouse, model)
               med_json["{}".format(warehouse)] = GradientBoosting(date_time_departure, real_warehouse, 'delivery',values, 'DEL_TRUE-DEPARTURE')
               med_json_eta["{}".format(warehouse)]= (datetime.utcfromtimestamp(time.mktime(datetime.strptime(med_json["{}".format(warehouse)], '%d/%m/%Y %H:%M:%S').timetuple()) + random.randrange(-129600, 129600, 1)).strftime('%d/%m/%Y %H:%M:%S'))#.strftime('%d/%m/%Y %H:%M:%S')
            #The same process as above made for Wavenet model
        print(type(med_json))
        json_result.append(med_json)
        pick_result.append(med_json_pick)
        json_eta.append(med_json_eta)
    print(json_result)
    print(type(json_result[0]))
    pd_result = pd.DataFrame(json_result)
    pd_pick = pd.DataFrame(pick_result)
    pd_eta = pd.DataFrame(json_eta)
    print(pd_result)
    mydb = getDatabaseConnection()
    dbcur = mydb.cursor()
    stmt = "SHOW TABLES LIKE 'result'"
    dbcur.execute(stmt)
    existment = dbcur.fetchone()
    pick_stmt = "SHOW TABLES LIKE 'pick_result'"
    dbcur.execute(pick_stmt)
    pick_existment = dbcur.fetchone()
    stmt_temp = "SHOW TABLES LIKE 'temp_result'"
    dbcur.execute(stmt_temp)
    temp_existment = dbcur.fetchone()
    print(temp_existment)
    stmt_temp_pick = "SHOW TABLES LIKE 'pick_temp_result'"
    dbcur.execute(stmt_temp_pick)
    pick_temp_existment = dbcur.fetchone()
    print(pick_temp_existment)
    stmt_eta_temp = "SHOW TABLES LIKE 'eta_result'"
    dbcur.execute(stmt_eta_temp)
    #temp_eta_existment = dbcur.fetchone()
    eta_existment = dbcur.fetchone()
    print(eta_existment)
    stmt_temp_eta = "SHOW TABLES LIKE 'eta_temp_result'"
    dbcur.execute(stmt_temp_eta)
    eta_temp_existment = dbcur.fetchone()
    result_list = []
    pick_result_list = []
    eta_result_list = []
    result_id = []
    pick_result_id = []
    eta_id = []
    eta_result_id = []
    if temp_existment:
        #pass
        query = "DROP TABLE temp_result"
        dbcur.execute(query)
    dbcur.execute("CREATE TABLE temp_result (id VARCHAR(255), date VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255), WH5 VARCHAR(255))")
    if pick_temp_existment:
        #pass
        query = "DROP TABLE pick_temp_result"
        dbcur.execute(query)
    dbcur.execute("CREATE TABLE pick_temp_result (id VARCHAR(255), date VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255), WH5 VARCHAR(255))")
    if eta_temp_existment:
        query = "DROP TABLE eta_temp_result"
        dbcur.execute(query)
    dbcur.execute("CREATE TABLE eta_temp_result (id VARCHAR(255), date VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255), WH5 VARCHAR(255))")
    for i in range (0, len(pd_result)):
        query = "INSERT INTO temp_result (id,date,WH1,WH2,WH3,WH4,WH5) " 
        query += " VALUES (%s, %s, %s, %s, %s, %s, %s)"
        #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
        values = (pd_result.iloc[i]['id'], order, pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'])
        print(values)
        dbcur.execute(query, values)
        mydb.commit()
    for i in range (0, len(pd_pick)):
        query = "INSERT INTO pick_temp_result(id, date,WH1,WH2,WH3,WH4,WH5)" 
        query += " VALUES (%s,%s, %s, %s, %s, %s, %s)"
        #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
        values = (pd_pick.iloc[i]['id'], order, pd_pick.iloc[i]['WH1'], pd_pick.iloc[i]['WH2'], pd_pick.iloc[i]['WH3'], pd_pick.iloc[i]['WH4'], pd_pick.iloc[i]['WH5'])
        print(values)
        dbcur.execute(query, values)
        mydb.commit()
    for i in range (0, len(pd_eta)):
        query = "INSERT INTO eta_temp_result(id, date,WH1,WH2,WH3,WH4,WH5)" 
        query += " VALUES (%s,%s, %s, %s, %s, %s, %s)"
        #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
        values = (pd_eta.iloc[i]['id'], order, pd_eta.iloc[i]['WH1'], pd_eta.iloc[i]['WH2'], pd_eta.iloc[i]['WH3'], pd_eta.iloc[i]['WH4'], pd_eta.iloc[i]['WH5'])
        print(values)
        dbcur.execute(query, values)
        mydb.commit()
    if existment:
        #pass
        query = "SELECT DISTINCT shipment, id FROM result"#"DROP TABLE result"
        ###dbcur.execute(query)
        # fetch all the matching rows 
        dbcur.execute(query)
        new_result = dbcur.fetchall()

        
        # loop through the rows
        for row in new_result:
            print(row)
            result_list.append(tuple((row[0], row[1])))
            print("\n")
    else:
        dbcur.execute("CREATE TABLE result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255), WH5 VARCHAR(255))")
    
    if pick_existment:
        #pass
        query = "SELECT DISTINCT shipment, id FROM pick_result"#"DROP TABLE result"
        ###dbcur.execute(query)
        # fetch all the matching rows 
        dbcur.execute(query)
        new_pick_result = dbcur.fetchall()

        
        # loop through the rows
        for row in new_pick_result:
            print(row)
            pick_result_list.append(tuple((row[0], row[1])))
            print("\n")
    else:
        dbcur.execute("CREATE TABLE pick_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255), WH5 VARCHAR(255))")
    
    if eta_existment:
         query = "SELECT DISTINCT shipment, id FROM eta_result"
         dbcur.execute(query)
         new_eta_result = dbcur.fetchall()
         # loop through the rows
         for row in new_pick_result:
            print(row)
            eta_result_list.append(tuple((row[0], row[1])))
            print("\n")
    else:
        dbcur.execute("CREATE TABLE eta_result (shipment VARCHAR(255),date VARCHAR(255),id VARCHAR(255), WH1 VARCHAR(255), WH2 VARCHAR(255), WH3 VARCHAR(255), WH4 VARCHAR(255), WH5 VARCHAR(255))")
    
    print(pick_result_list)
    print(result_list)
    print(eta_result_list)
    #sql_result   = pd_result.to_sql('result', mydb, if_exists= 'fail')
    #print(sql_result)
    print('++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++')
    print(pd_pick)
    print(pd_result)
    print(pd_eta)
    for i in range (0, len(pd_result)):
        print(tuple((pd_result.iloc[i]['shipment'],pd_result.iloc[i]['id'])))
        print(result_list)
        print('HAHAHAHAHAHA')
        if tuple((pd_result.iloc[i]['shipment'],pd_result.iloc[i]['id'])) in result_list:
                query = "SELECT DISTINCT date FROM result"
                dbcur.execute(query)
                date_result = dbcur.fetchall()
                print(date_result)
                print('************************')
                ##query = "INSERT INTO result (shipment,id,WH1,WH2,WH3,WH4,WH5) " 
                ##query += " VALUES (%s,%s, %s, %s, %s, %s, %s)"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                ##values = (pd_result.iloc[i]['shipment'], pd_result.iloc[i]['id'], pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'])
                ##print(values)
                ##dbcur.execute(query, values)
                ##mydb.commit()
                print(tuple((order,)))
                if tuple((order,)) in date_result:
                    query = "UPDATE result SET shipment = %s,id = %s, WH1 = %s, WH2 = %s, WH3 = %s, WH4 = %s, WH5 = %s WHERE shipment = %s AND id = %s" 
                    values = (pd_result.iloc[i]['shipment'], pd_result.iloc[i]['id'], pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'],pd_result.iloc[i]['shipment'], pd_result.iloc[i]['id'])
                    print(query)
                    print(values)
                    dbcur.execute(query, values)
                    mydb.commit()
                #r'''
                else:
                    query = "INSERT INTO result (shipment,date,id,WH1,WH2,WH3,WH4,WH5) " 
                    query += " VALUES (%s,%s,%s, %s, %s, %s, %s, %s)"
                    #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                    values = (pd_result.iloc[i]['shipment'], order,pd_result.iloc[i]['id'], pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'])
                    print(values)
                    dbcur.execute(query, values)
                    mydb.commit()
                #'''
        else:
                query = "INSERT INTO result (shipment,date,id,WH1,WH2,WH3,WH4,WH5) " 
                query += " VALUES (%s,%s,%s, %s, %s, %s, %s, %s)"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                values = (pd_result.iloc[i]['shipment'], order,pd_result.iloc[i]['id'], pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'])
                print(values)
                dbcur.execute(query, values)
                mydb.commit()
    for i in range (0, len(pd_pick)):
            #print(pd_pick.iloc[i]['id'])
            #print(pd_pick)
            if tuple((pd_pick.iloc[i]['shipment'],pd_pick.iloc[i]['id'])) in pick_result_list:
                query = "SELECT DISTINCT date FROM pick_result"
                dbcur.execute(query)
                date_result = dbcur.fetchall()
                ##query = "INSERT INTO result (shipment,id,WH1,WH2,WH3,WH4,WH5) " 
                ##query += " VALUES (%s,%s, %s, %s, %s, %s, %s)"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                ##values = (pd_result.iloc[i]['shipment'], pd_result.iloc[i]['id'], pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'])
                ##print(values)
                ##dbcur.execute(query, values)
                ##mydb.commit()
                ##if order in date_result:
                if tuple((order,)) in date_result:
                    query = "UPDATE pick_result SET shipment = %s,date=%s,id = %s, WH1 = %s, WH2 = %s, WH3 = %s, WH4 = %s, WH5 = %s WHERE shipment = %s AND id = %s" 
                    values = (pd_pick.iloc[i]['shipment'],order, pd_pick.iloc[i]['id'], pd_pick.iloc[i]['WH1'], pd_pick.iloc[i]['WH2'], pd_pick.iloc[i]['WH3'], pd_pick.iloc[i]['WH4'], pd_pick.iloc[i]['WH5'],pd_pick.iloc[i]['shipment'], pd_pick.iloc[i]['id'])
                    print(query)
                    print(values)
                    dbcur.execute(query, values)
                    mydb.commit()
                else:
                    query = "INSERT INTO pick_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5) " 
                    query += " VALUES (%s,%s,%s, %s, %s, %s, %s, %s)"
                    #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                    values = (pd_pick.iloc[i]['shipment'], order, pd_pick.iloc[i]['id'], pd_pick.iloc[i]['WH1'], pd_pick.iloc[i]['WH2'], pd_pick.iloc[i]['WH3'], pd_pick.iloc[i]['WH4'], pd_pick.iloc[i]['WH5'])
                    print(values)
                    dbcur.execute(query, values)
                    mydb.commit()
            else:
                query = "INSERT INTO pick_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5) " 
                query += " VALUES (%s,%s,%s, %s, %s, %s, %s, %s)"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                values = (pd_pick.iloc[i]['shipment'], order, pd_pick.iloc[i]['id'], pd_pick.iloc[i]['WH1'], pd_pick.iloc[i]['WH2'], pd_pick.iloc[i]['WH3'], pd_pick.iloc[i]['WH4'], pd_pick.iloc[i]['WH5'])
                print(values)
                dbcur.execute(query, values)
                mydb.commit()

    for i in range (0, len(pd_eta)):
            #print(pd_eta.iloc[i]['id'])
            #print(pd_eta)
            if tuple((pd_eta.iloc[i]['shipment'],pd_eta.iloc[i]['id'])) in eta_result_list:
                query = "SELECT DISTINCT date FROM eta_result"
                dbcur.execute(query)
                date_result = dbcur.fetchall()
                ##query = "INSERT INTO result (shipment,id,WH1,WH2,WH3,WH4,WH5) " 
                ##query += " VALUES (%s,%s, %s, %s, %s, %s, %s)"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                ##values = (pd_result.iloc[i]['shipment'], pd_result.iloc[i]['id'], pd_result.iloc[i]['WH1'], pd_result.iloc[i]['WH2'], pd_result.iloc[i]['WH3'], pd_result.iloc[i]['WH4'], pd_result.iloc[i]['WH5'])
                ##print(values)
                ##dbcur.execute(query, values)
                ##mydb.commit()
                ##if order in date_result:
                if tuple((order,)) in date_result:
                    query = "UPDATE eta_result SET shipment = %s,date=%s,id = %s, WH1 = %s, WH2 = %s, WH3 = %s, WH4 = %s, WH5 = %s WHERE shipment = %s AND id = %s" 
                    values = (pd_eta.iloc[i]['shipment'],order, pd_eta.iloc[i]['id'], pd_eta.iloc[i]['WH1'], pd_eta.iloc[i]['WH2'], pd_eta.iloc[i]['WH3'], pd_eta.iloc[i]['WH4'], pd_eta.iloc[i]['WH5'],pd_eta.iloc[i]['shipment'], pd_eta.iloc[i]['id'])
                    print(query)
                    print(values)
                    dbcur.execute(query, values)
                    mydb.commit()
                else:
                    query = "INSERT INTO eta_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5) " 
                    query += " VALUES (%s,%s,%s, %s, %s, %s, %s, %s)"
                    #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                    values = (pd_eta.iloc[i]['shipment'], order, pd_eta.iloc[i]['id'], pd_eta.iloc[i]['WH1'], pd_eta.iloc[i]['WH2'], pd_eta.iloc[i]['WH3'], pd_eta.iloc[i]['WH4'], pd_eta.iloc[i]['WH5'])
                    print(values)
                    dbcur.execute(query, values)
                    mydb.commit()
            else:
                query = "INSERT INTO eta_result (shipment,date,id,WH1,WH2,WH3,WH4,WH5) " 
                query += " VALUES (%s,%s,%s, %s, %s, %s, %s, %s)"
                #values = ('Shipment 1', datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
                values = (pd_eta.iloc[i]['shipment'], order, pd_eta.iloc[i]['id'], pd_eta.iloc[i]['WH1'], pd_eta.iloc[i]['WH2'], pd_eta.iloc[i]['WH3'], pd_eta.iloc[i]['WH4'], pd_eta.iloc[i]['WH5'])
                print(values)
                dbcur.execute(query, values)
                mydb.commit()
    
    return jsonify({"The models are trained":200})



@app.route("/GetResult", methods=['GET'])
#In this function the nmrse results are created
def json_result():
    mydb = getDatabaseConnection()
    cursor = mydb.cursor()
    select_query = "SELECT * FROM temp_result"
    cursor.execute(select_query)
    all_assets = cursor.fetchall()
    print(all_assets)
    print(config['columns']['temp_result'])
    init_data = pd.DataFrame(data = all_assets,columns = config['columns']['temp_result'])#.iloc[:,1:]
    print(init_data)
    result = init_data.to_json(orient="records")
    parsed = json.loads(result)
    print(parsed)
    print(type(parsed[0]))
    #json.dumps(parsed, indent=4)  

    pick_query = "SELECT * FROM pick_temp_result"
    cursor.execute(pick_query)
    all_assets = cursor.fetchall()

    init_data = pd.DataFrame(data = all_assets,columns = config['columns']['temp_result'])#.iloc[:,1:]
    print(init_data)
    result = init_data.to_json(orient="records")
    pick = json.loads(result)
    print(pick)
    print(type(pick[0]))
    #json.dumps(parsed, indent=4)  

    eta_query = "SELECT * FROM eta_temp_result"
    cursor.execute(eta_query)
    all_assets = cursor.fetchall()

    init_data = pd.DataFrame(data = all_assets,columns = config['columns']['temp_result'])#.iloc[:,1:]
    print(init_data)
    result = init_data.to_json(orient="records")
    eta = json.loads(result)
    print(eta)
    print(type(eta[0]))
    #json.dumps(parsed, indent=4)  

    nrmse_query = "SELECT * FROM nrmse_delivery"
    cursor.execute(nrmse_query)
    all_assets = cursor.fetchall()
    nrmse_data = pd.DataFrame(data = all_assets,columns = config['columns']['nrmse'])#.iloc[:,1:]
    print(nrmse_data)
    nrmse_result = nrmse_data.to_json(orient="records")
    nrmse_parsed = json.loads(nrmse_result)
    print(nrmse_parsed )
    print(type(nrmse_parsed [0]))
    #final_table = jsonify(parsed) + jsonify(nrmse_parsed)
    #print(final_table)

    nrmse_dep_query = "SELECT * FROM nrmse_departure"
    cursor.execute(nrmse_dep_query)
    all_assets = cursor.fetchall()
    nrmse_dep_data = pd.DataFrame(data = all_assets,columns = config['columns']['nrmse'])#.iloc[:,1:]
    print(nrmse_dep_data)
    nrmse_dep_result = nrmse_dep_data.to_json(orient="records")
    nrmse_dep_parsed = json.loads(nrmse_dep_result)
    print(nrmse_dep_parsed)
    print(type(nrmse_dep_parsed [0]))
    #final_table = jsonify(parsed) + jsonify(nrmse_parsed)
    #print(final_table)


    nrmse_eta_query = "SELECT * FROM nrmse_eta_delivery"
    cursor.execute(nrmse_eta_query)
    all_assets = cursor.fetchall()
    nrmse_eta_data = pd.DataFrame(data = all_assets,columns = config['columns']['nrmse'])#.iloc[:,1:]
    print(nrmse_eta_data)
    nrmse_eta_result = nrmse_eta_data.to_json(orient="records")
    nrmse_eta_parsed = json.loads(nrmse_eta_result)
    print(nrmse_eta_parsed )
    print(type(nrmse_eta_parsed [0]))
    #final_table = jsonify(parsed) + jsonify(nrmse_parsed)
    #print(final_table)
    return jsonify(pick,nrmse_dep_parsed,eta,nrmse_eta_parsed,parsed,nrmse_parsed)
    
    #return jsonify([jsonify(parsed), jsonify(nrmse_parsed)]) #Prediction is estimated', 200

#Get the values corresponding to 3 first columns of DEPARTURE-ORDER dataframe named ORDER_MONTH_OF_YEAR,ORDER_DAY_OF_MONTH,ORDER_DAY_OF_WEEK
def departure_dates(date, target,  warehouse, model):
    date_time_obj = datetime.strptime(date, '%d/%m/%Y %H:%M:%S')
    weekday = np.int64(date_time_obj.weekday())# It is estimated ORDER_DAY_OF_WEEK
    day = np.int64(date_time_obj.day)#It is estimated ORDER_DAY_OF_MONTH
    month = np.int64(date_time_obj.month)#It is estimated ORDER_MONTH_OF_YEAR

    columns = config['columns']['{}'.format(target)]#[:-1]
    values = [0] * (len(columns)-2)# it is subtracted the id column and the target column (DEPARTURE-ORDER)
    if model == 'WN':
        values = [0] * (len(columns)-1)
    values[0] = month # assign the right value on value list in the item position corresponding to ORDER_MONTH_OF_YEAR
    values[1] = day#assign the right value on value list in the item position corresponding to ORDER_DAY_OF_MONTH
    values[2] = weekday#assign the right value on value list in the item position corresponding to ORDER_DAY_OF_WEEK
    pos = columns.index('{}'.format(warehouse))-1 #FInd the corresponding item in values list for the warehouse
    values[pos] = 1#assign the value 1 to the corresponding item for the warehouse
    return values, date_time_obj

#Get the values corresponding to 2-8 first columns of DEL_TRUE-DEPARTURE dataframe named ORDER_MONTH_OF_YEAR,DEPARTURE_MONTH_OF_YEAR,ORDER_DAY_OF_MONTH,DEPARTURE_DAY_OF_MONTH,ORDER_DAY_OF_WEEK,DEPARTURE_DAY_OF_WEEK
def delivery_dates(order, departure, target, warehouse, model):
    date_time_order = datetime.strptime(order, '%d/%m/%Y %H:%M:%S')#Convert the string format of order datetime to datetime object
    date_time_departure = datetime.strptime(departure, '%d/%m/%Y %H:%M:%S')#Convert the string format of departure datetime to datetime object
    duration = date_time_departure - date_time_order                         
    duration_in_s = duration.total_seconds()# The difference between departure and order datetime in seconda 
    print(duration_in_s) 
    weekday_order = np.int64(date_time_order.weekday())# It is estimated ORDER_DAY_OF_WEEK
    day_order = np.int64(date_time_order.day)#It is estimated ORDER_DAY_OF_MONTH
    month_order = np.int64(date_time_order.month)#It is estimated ORDER_MONTH_OF_YEAR
    weekday_departure = np.int64(date_time_departure.weekday())#It is estimated DEPARTURE_DAY_OF_WEEK
    day_departure = np.int64(date_time_departure.day)#It is estimated DEPARTURE_DAY_OF_MONTH
    month_departure = np.int64(date_time_departure.month)##It is estimated DEPARTURE_DAY_OF_MONTH
    columns = config['columns']['{}'.format(target)]#[:-1]
    values = [0] * (len(columns) - 2)
    if model == 'WN':
        values = [0] * (len(columns)-1)
    #The DEPARTURE-ORDER is the first column of DEL_TRUE-DEPARTURE dataframe and we estimate the differnce in hours for the new data point
    values[0] = int(divmod(duration_in_s, 3600)[0])  
    values[1] = month_order# assign the right value on value list in the item position corresponding to ORDER_MONTH_OF_YEAR
    values[2] = month_departure# assign the right value on value list in the item position corresponding to DEPARTURE_MONTH_OF_YEAR
    values[3] = day_order#assign the right value on value list in the item position corresponding to ORDER_DAY_OF_MONTH
    values[4] = day_departure#assign the right value on value list in the item position corresponding to DEPARTURE_DAY_OF_MONTH
    values[5] = weekday_order#assign the right value on value list in the item position corresponding to ORDER_DAY_OF_WEEK
    values[6] = weekday_departure#assign the right value on value list in the item position corresponding to DEPARTURE_DAY_OF_WEEK
    pos = columns.index('{}'.format(warehouse))-1#FInd the corresponding item in values list for the warehouse
    values[pos] = 1#assign the value 1 to the corresponding item for the warehouse
    return values, date_time_departure

#The algorithmic process for wavenet model
def wavenet(date_time, warehouse,target, values):
    initial_path = os.getcwd()
    window_size = int(config[target][warehouse ]['windows_size'])
    time_step = int(config[target][warehouse ]['timestep'])
    try:
        result = []
        mydb = getDatabaseConnection()
        cursor = mydb.cursor()
        select_query = "SELECT * FROM {}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), target.split('-')[0].lower(), target.split('-')[1].lower())
        cursor.execute(select_query)
        all_assets = cursor.fetchall()
        init_data = pd.DataFrame(data = all_assets,columns = config['columns']['{}'.format(target)] ).iloc[:,1:]
        print(init_data)
        model_dir = os.path.join(os.getcwd(), "wavenet_models")
        os.chdir(model_dir)
        #model3 = tf.keras.models.load_model('{} {}.pkl'.format(warehouse, target))
        model3 = keras.models.load_model('{} {}.pkl'.format(warehouse, target))
        print('HAHAHAHAHAHAHA')
        model3.summary()
        os.chdir(initial_path)
        columns = list(init_data.columns)#[:-1]
        df1 = pd.DataFrame ([values], columns = columns)
        df = pd.concat([init_data[columns], df1]).reset_index(drop=True)
        print(df)
        os.chdir(initial_path)
        scaler_dir = os.path.join(os.getcwd(), "scalers")
        os.chdir(scaler_dir)
        scaler = load(open(warehouse + "_" + target+ '_scaler.pkl', 'rb'))
        os.chdir(initial_path)
        X = scaler.fit(df).transform(df)
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
        init_prediction = abs(preds[-1][0][0])
        min = df.iloc[:,-1].min()
        max =  df.iloc[:,-1].max()
        final_prediction = (init_prediction * max + (1 - init_prediction) * min)
        estimated_delivery  = date_time + timedelta(hours=int(final_prediction))
        prediction = estimated_delivery.strftime('%d/%m/%Y %H:%M:%S')
        print(prediction)
        print('Final Time as string object: ', prediction )
    except Exception as e:
        prediction = None
        print("====================" + str(e) + "====================")
    finally:
        cursor.close()
        mydb.close()
    return prediction
   
#The algorithmic process for Random Forest model       
def RandomForest(date_time, warehouse, values, folder_target):
    initial_path = os.getcwd()
    model_dir =  os.path.join(os.getcwd(), "rf_models_{}".format(folder_target))
    os.chdir(model_dir)
    with open('rf_{}'.format(warehouse),"rb") as f:
        randomForestRegressor = pickle.load(f)
        os.chdir(initial_path)
        x = np.array(values)
        x = x.reshape(1, -1)
        final_prediction = randomForestRegressor.predict(x)
        estimated_delivery  = date_time + timedelta(hours=int(final_prediction))
        prediction = estimated_delivery.strftime('%d/%m/%Y %H:%M:%S')
        #datetime. strptime(date_time_str, '%d/%m/%Y %H:%M:%S')
        print(prediction)
        print('Final Time as string object: ', prediction )
    return prediction

def bagging(date_time, warehouse,target, values, all_target):
    initial_path = os.getcwd()
    
    try:
        result = []
        mydb = getDatabaseConnection()
        print('******************')
        print(warehouse)
        print(target)
        print("{}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), all_target.split('-')[0].lower(), all_target.split('-')[1].lower()))
        cursor = mydb.cursor()
        select_query = "SELECT * FROM {}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), all_target.split('-')[0].lower(), all_target.split('-')[1].lower())
        print(select_query)
        cursor.execute(select_query)
        all_assets = cursor.fetchall()
        init_data = pd.DataFrame(data = all_assets,columns = config['columns']['{}'.format(all_target)] ).iloc[:,1:]
        print('INIT DATA')
        print(init_data)
        model_dir = os.path.join(os.getcwd(), "bagging_model_results","models","b_models_{}".format(target))
        os.chdir(model_dir)
        with open('b_{}_{}.pkl'.format(warehouse.split()[0], warehouse.split()[1]),"rb") as f:
            baggingRegr = pickle.load(f)
        os.chdir(initial_path)
        columns = list(init_data.columns)[:-1]
        df1 = pd.DataFrame ([values], columns = columns)
        df = pd.concat([init_data[columns], df1]).reset_index(drop=True)
        print('This is df')
        print(df)
        os.chdir(initial_path)
        scaler_dir = os.path.join(os.getcwd(), "bagging_model_results", "scalers", "b_models_{}".format(target))#C:\Users\skoumpmi\Downloads\project\project\bagging_model_results\scalers\b_models_delivery
        os.chdir(scaler_dir)
        scaler = pickle.load(open('b_{}_{}.pkl'.format(warehouse.split()[0],warehouse.split()[1]),"rb"))
        os.chdir(initial_path)
        X = scaler.fit(df).transform(df)
        #X = X.iloc[-1, :]
        X = np.array(X)[-1]
        X = X.reshape(1, -1)
        print(X.shape)
        init_prediction = baggingRegr.predict(X)
        print('init_prediction_is')
        print(init_prediction[0])
        min = init_data.iloc[:,-1].min()
        max =  init_data.iloc[:,-1].max()
        print(min)
        print(max)
        final_prediction = (init_prediction[0] * max + (1 - init_prediction[0]) * min)
        print(final_prediction)
        print(date_time)
        estimated_delivery  = date_time + timedelta(hours=int(final_prediction))
        print("========================================")
        print(estimated_delivery)
        prediction = estimated_delivery.strftime('%d/%m/%Y %H:%M:%S')
        print(prediction)
        print('Final Time as string object: ', prediction )
        print("========================================")
    except Exception as e:
        prediction = None
        print("====================" + str(e) + "====================")
    finally:
        cursor.close()
        mydb.close()
    return prediction
   
def GradientBoosting(date_time, warehouse,target, values, all_target):
    initial_path = os.getcwd()
    
    try:
        result = []
        mydb = getDatabaseConnection()
        print('******************')
        print(warehouse)
        print(target)
        print("{}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), all_target.split('-')[0].lower(), all_target.split('-')[1].lower()))
        cursor = mydb.cursor()
        select_query = "SELECT * FROM {}_{}_{}_{}".format(warehouse.split()[0].lower(), warehouse.split()[1].lower(), all_target.split('-')[0].lower(), all_target.split('-')[1].lower())
        print(select_query)
        cursor.execute(select_query)
        all_assets = cursor.fetchall()
        init_data = pd.DataFrame(data = all_assets,columns = config['columns']['{}'.format(all_target)] ).iloc[:,1:]
        print('INIT DATA')
        print(init_data)
        model_dir = os.path.join(os.getcwd(), "gradientboosting_model_results","models","gb_models_{}".format(target))
        os.chdir(model_dir)
        with open('gb_{}_{}.pkl'.format(warehouse.split()[0], warehouse.split()[1]),"rb") as f:
            GBRegr = pickle.load(f)
        os.chdir(initial_path)
        columns = list(init_data.columns)[:-1]
        df1 = pd.DataFrame ([values], columns = columns)
        df = pd.concat([init_data[columns], df1]).reset_index(drop=True)
        print('This is df')
        print(df)
        os.chdir(initial_path)
        scaler_dir = os.path.join(os.getcwd(), "gradientboosting_model_results", "scalers", "gb_models_{}".format(target))#C:\Users\skoumpmi\Downloads\project\project\bagging_model_results\scalers\b_models_delivery
        os.chdir(scaler_dir)
        scaler = pickle.load(open('gb_{}_{}.pkl'.format(warehouse.split()[0],warehouse.split()[1]),"rb"))
        os.chdir(initial_path)
        X = scaler.fit(df).transform(df)
        #X = X.iloc[-1, :]
        X = np.array(X)[-1]
        X = X.reshape(1, -1)
        print(X.shape)
        init_prediction = GBRegr.predict(X)
        print('init_prediction_is')
        print(init_prediction[0])
        min = init_data.iloc[:,-1].min()
        max =  init_data.iloc[:,-1].max()
        print(min)
        print(max)
        final_prediction = (init_prediction[0] * max + (1 - init_prediction[0]) * min)
        print(final_prediction)
        print(date_time)
        estimated_delivery  = date_time + timedelta(hours=int(final_prediction))
        print("========================================")
        print(estimated_delivery)
        prediction = estimated_delivery.strftime('%d/%m/%Y %H:%M:%S')
        print(prediction)
        print('Final Time as string object: ', prediction )
        print("========================================")
    except Exception as e:
        prediction = None
        print("====================" + str(e) + "====================")
    finally:
        cursor.close()
        mydb.close()
    return prediction


#Get the database credentials in order to connect
def getDatabaseConnection():
        return mySQL.connect(host=config['database']['host'], user=config['database']['user'], passwd=config['database']['passwd'], db=config['database']['db'], charset=config['database']['charset'], auth_plugin='mysql_native_password')




app.run(host='localhost', port=5003)

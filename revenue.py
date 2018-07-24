from flask import Flask
from flask import request
import json
import requests
import unicodedata
from datetime import datetime, timedelta
import numpy as np
import MySQLdb as dbapi
import sys
import csv
import mysql.connector
import pandas as pd
import MySQLdb as dbapi
import mysql
import sys
import csv
from pandas import Series
from matplotlib import pyplot
import mysql.connector
import pandas as pd
from matplotlib import pyplot
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy
import requests
import calendar
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy
import operator
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

# json_data = json.dumps(array2)
# json_data1=json.dumps(sorted_next)

app = Flask(__name__)


@app.route("/")
def hello():
    fromDate = request.args.get('from')
    toDate = request.args.get('to')
    siteId = request.args.get('site_id')
    firstDate = connect(siteId)
    array2 = filterDates(fromDate, toDate, firstDate)
    return json.dumps(array2)


# def lessdata():
#     if (length<60):
#           return {'status' : 'error', 'message' : 'Insuffient data'}


def filterDates(fromDate, toDate, firstDate):
    df_2 = pd.read_csv("revenue1.csv")
    length = len(df_2)
    print
    (length)
    if (length < 64):
        return {'status': 'error', 'message': 'Insuffient data'}
    fromDate = unicodedata.normalize('NFKD', fromDate).encode('ascii', 'ignore')
    toDate = unicodedata.normalize('NFKD', toDate).encode('ascii', 'ignore')
    fromDate = datetime.strptime(fromDate.decode('ascii'), '%Y-%m-%d')
    toDate = datetime.strptime(toDate.decode('ascii'), '%Y-%m-%d')
    firstDate1 = datetime.strptime(firstDate, '%Y-%m-%d')
    diff1 = int(((fromDate - firstDate1).total_seconds()) / timedelta(days=1).total_seconds())
    diff2 = int(((toDate - firstDate1).total_seconds()) / timedelta(days=1).total_seconds())
    if (diff2 + 1 > 40):
        return {'status': 'error', 'message': 'Date limit exceeded'}
    array2 = getData(diff1, diff2)
    return array2


# ------------------------------------------------------------------------------


def connect(siteId):
    conn = mysql.connector.connect(
        user='root',
        password='',
        host='127.0.0.1',
        database='production_db')

    cur = conn.cursor(buffered=True)
    #     siteId = '63'
    orderStatus = 'CANCELLED'
    sql = """
    SELECT CAST(orders.`created_at` AS DATE) AS order_date, SUM(total) AS order_total FROM orders 
    INNER JOIN sites ON orders.`site_id` = sites.id 
    INNER JOIN users ON orders.`user_id` = users.`id`
    WHERE orders.site_id = """ + siteId + """ AND order_number > 0 AND users.email NOT LIKE '%@costprize.com' 
    AND users.email NOT LIKE '%@gito.me' AND orders.`status` != 'CART_INPROGRESS' AND orders.`status` != '""" + orderStatus + """'
    GROUP BY CAST(orders.`created_at` AS DATE) ORDER BY order_date ASC
    """
    # cur=conn.cursor()
    cur.execute(sql)
    result = cur.fetchall()

    csvFile = open('revenue1.csv', 'w')
    c = csv.writer(csvFile)
    for x in result:
        c.writerow(x)
    csvFile.close()
    return readcsv()


# ------------------------------------------------------------------------------
dateList = []


def readcsv():
    dateArr = []
    data_frame = pd.read_csv('revenue1.csv')
    date = data_frame.iloc[:, [0]]
    dateArr = date.values.tolist()
    lastDate = dateArr[-1][0]
    format_str = '%Y-%m-%d'  # The format
    datetime_obj = datetime.strptime(lastDate, format_str)
    firstDate = (datetime_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    del dateList[:]
    dateList2 = []
    for i in range(50):
        datetime_obj += timedelta(days=1)
        dateList.append(datetime_obj.strftime('%Y-%m-%d'))
    #     for first_10date in dateList:
    #         print first_10date
    #     for i in range(50):
    #         datetime_obj += timedelta(days=1)
    #         dateList2.append(datetime_obj.strftime('%Y-%m-%d'))
    #     for second_10date in dateList2:
    #         print (second_10date)
    #     for i in range(0,50):
    #         print dateList[i]
    #     print dateList
    return firstDate


#-------------------------------------------------------------------------------------------

result = []


def getResult():
    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # load dataset
    series = Series.from_csv('revenue1.csv', header=None)

    # seasonal difference
    X = series.values
    X = X.astype('float32')
    days_in_month = 30
    differenced = difference(X, days_in_month)

    # fit model
    best_cfg = arima()
    model = ARIMA(differenced, order=(best_cfg))
    model_fit = model.fit(disp=0)

    # multi-step out-of-sample forecast
    start_index = len(differenced)
    end_index = start_index + 50
    forecast = model_fit.predict(start=start_index, end=end_index)

    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    del result[:]
    #     print "Revenue for first 10 days"
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_month)
        # print ('day %d:= %d' % (day, inverted))
        result1 = "%f" % (inverted)
        result.append(result1)
        history.append(inverted)
        day += 1


#     for first_10revenue in result[0:50]:
#         print first_10revenue

# #------------------------------------------------------------------------------

# create a differenced series
def arima():
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # evaluate an ARIMA model for a given order (p,d,q) and return RMSE
    def evaluate_arima_model(X, arima_order):
        # prepare training dataset
        X = X.astype('float32')
        train_size = int(len(X) * 0.50)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            # difference data
            days_in_month = 30
            diff = difference(history, days_in_month)
            model = ARIMA(diff, order=arima_order)
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            yhat = inverse_difference(history, yhat, days_in_month)
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mse)
        return rmse

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    #                     print order
                    try:
                        mse = evaluate_arima_model(dataset, order)
                        print
                        mse
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s RMSE=%.2f' % (order, mse))
                    except:
                        continue
        print('Best ARIMA%s RMSE=%.2f' % (best_cfg, best_score))
        return best_cfg

    series = Series.from_csv('revenue1.csv')
    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 4)
    warnings.filterwarnings("ignore")
    best_cfg = evaluate_models(series.values, p_values, d_values, q_values)
    return best_cfg


def getData(diff1, diff2):
    getResult()
    results = ""
    result_dict = {}
    array1 = []
    #sorted_y = {}
    array2 = []
    for i in range(diff1, diff2 + 1):
        array1.append(dateList[i] + ' = ' + 'INR' + result[i])
        for first in array1:
            result_dict.update({dateList[i]: result[i]})
    x = result_dict
    sorted_y = sorted(x.items(), key=operator.itemgetter(0))
    for i in range(len(sorted_y)):
        array2.append({sorted_y[i][0]: sorted_y[i][1]})
    return array2


#     print array2
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8049)
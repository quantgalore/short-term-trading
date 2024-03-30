# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import mysql.connector
import sqlalchemy
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from sklearn.ensemble import GradientBoostingClassifier

def binarizer(value):
    
    if value > 0: return 1 
    else: return 0

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')

trading_dates = calendar.schedule(start_date = (datetime.today()-timedelta(days=10)), end_date = (datetime.today())).index.strftime("%Y-%m-%d").values
today = trading_dates[-1]

benchmark_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2017-01-01/{today}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["pct_change"] = round(benchmark_data["c"].pct_change()*100,2)

##

tickers = np.array(["SPY"])

prediction_list = []
times = []

for ticker in tickers:
    
    try:
        
        start_time = datetime.now()
        
        ticker_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2017-01-01/{today}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        ticker_data.index = pd.to_datetime(ticker_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        ticker_data["pct_change"] = round(ticker_data["c"].pct_change()*100,2)
        ticker_data["3_mo_avg"] = ticker_data["c"].rolling(window=63).mean()
        ticker_data["6_mo_avg"] = ticker_data["c"].rolling(window=126).mean()
        ticker_data['regime'] = ticker_data.apply(lambda row: 1 if (row['3_mo_avg'] > row['6_mo_avg']) else 0, axis=1)
        
        ticker_data = ticker_data[["c", "pct_change", "3_mo_avg", "6_mo_avg", "regime"]]
        
        ticker_data = ticker_data.dropna()
        
        if len(ticker_data) < 504:
            continue
        
        ticker_data_list = []
        
        for date in ticker_data.index.values[30:]:
    
            prior_30 = ticker_data[ticker_data.index < pd.to_datetime(date).strftime("%Y-%m-%d")].copy().tail(30)
            benchmark_prior_30 = benchmark_data[benchmark_data.index < pd.to_datetime(date).strftime("%Y-%m-%d")].copy().tail(30)
            
            historical_correlation = round(np.corrcoef(x=benchmark_prior_30["pct_change"].values, y=prior_30["pct_change"].values)[1][0]*100,2)
            
            avg_return = prior_30["pct_change"].mean()
            median_return = prior_30["pct_change"].median()
            return_std = prior_30["pct_change"].std()
            
            regime = prior_30["regime"].iloc[-1]
            
            date_data = ticker_data[ticker_data.index ==  pd.to_datetime(date).strftime("%Y-%m-%d")].copy()
            
            last_return = prior_30["pct_change"].iloc[-1]
            return_before_last = prior_30["pct_change"].iloc[-2]
            t_minus_3 = prior_30["pct_change"].iloc[-3]
            
            actual_return = date_data["pct_change"].iloc[0]
            
            return_dataframe = pd.DataFrame([{"date": pd.to_datetime(date), "ticker": ticker,  "avg_return": avg_return,
                                              "median_return": median_return, "return_before_last": return_before_last, "last_return": last_return,
                                              "t-3": t_minus_3,"return_std": return_std, "regime": regime, "actual_return": actual_return}])
            
            ticker_data_list.append(return_dataframe)
            
        full_ticker_data = pd.concat(ticker_data_list).reset_index(drop=True).set_index("date")
        full_ticker_data["actual_return"] = full_ticker_data["actual_return"].shift(-1)
        
        historical_data = full_ticker_data[full_ticker_data.index < today].copy().tail(504)
        
        X = historical_data.drop(["actual_return", "ticker"], axis = 1)
        Y = historical_data["actual_return"].apply(binarizer)
        
        GradientBoosing_Model = GradientBoostingClassifier().fit(X,Y)
        
        oos_data = full_ticker_data[full_ticker_data.index.date == pd.to_datetime(today).date()].copy()
        
        X_test = oos_data.drop(["actual_return", "ticker"], axis = 1)
        
        y_pred = GradientBoosing_Model.predict(X_test)
        y_pred_proba = GradientBoosing_Model.predict_proba(X_test)
        y_pred_proba = [y_pred_proba[0][y_pred[0]]]
        
        pred_df = pd.DataFrame({"date": oos_data.index[0], "pred": y_pred, "proba":y_pred_proba, "ticker": ticker,"correlation": historical_correlation,
                                "actual": oos_data["actual_return"].iloc[0]})
        
        prediction_list.append(pred_df)
        
        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(tickers==ticker)[0][0]/len(tickers))*100,2)
        iterations_remaining = len(tickers) - np.where(tickers==ticker)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
        
    except Exception as error:
        print(error, ticker)
        continue
    
# The day's predictions for your selected tickers
full_prediction_data = pd.concat(prediction_list)

# Separated into positive and negative predictions
positive_pred = full_prediction_data[full_prediction_data["pred"] == 1]
negative_pred = full_prediction_data[full_prediction_data["pred"] == 0]
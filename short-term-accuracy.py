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
from sklearn.neighbors import KNeighborsClassifier

def binarizer(value):
    
    if value > 0: return 1 
    else: return 0

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
trading_dates = calendar.schedule(start_date = (datetime.today()-timedelta(days=10)), end_date = (datetime.today())).index.strftime("%Y-%m-%d").values
today = trading_dates[-2]
##

# =============================================================================
# Dataset Builder
# =============================================================================

benchmark_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2015-01-01/{today}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
benchmark_data.index = pd.to_datetime(benchmark_data.index, unit="ms", utc=True).tz_convert("America/New_York")
benchmark_data["pct_change"] = round(benchmark_data["c"].pct_change()*100,2)

tickers_to_test = pd.read_sql("SELECT * FROM weekly_option_tickers", con = engine)["tickers"].values
# tickers_to_test = np.array(["TSLA"])

full_prediction_list = []

for ticker in tickers_to_test:
    
    try:
        
        full_ticker_data_list = []
        complete_ohlcv_list = []
        
        ticker_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2017-01-01/{datetime.today().strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        ticker_data.index = pd.to_datetime(ticker_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        ticker_data["pct_change"] = round(ticker_data["c"].pct_change()*100,2)
        ticker_data["3_mo_avg"] = ticker_data["c"].rolling(window=63).mean()
        ticker_data["6_mo_avg"] = ticker_data["c"].rolling(window=126).mean()
        ticker_data['regime'] = ticker_data.apply(lambda row: 1 if (row['3_mo_avg'] > row['6_mo_avg']) else 0, axis=1)
        
        ticker_data = ticker_data[["o", "c", "pct_change", "3_mo_avg", "6_mo_avg", "regime"]]
        
        ticker_data = ticker_data.dropna()
        
        if len(ticker_data) < 30:
            continue
        
        ticker_data["ticker"] = ticker
        
        complete_ohlcv_list.append(ticker_data)
        
        ticker_data_list = []
        
        for date in ticker_data.index.values[30:]:
        
            prior_30 = ticker_data[ticker_data.index < pd.to_datetime(date).strftime("%Y-%m-%d")].copy().tail(30)
            benchmark_prior_30 = benchmark_data[benchmark_data.index < pd.to_datetime(date).strftime("%Y-%m-%d")].copy().tail(30)
            
            historical_correlation = round(np.corrcoef(x=benchmark_prior_30["pct_change"].values, y=prior_30["pct_change"].values)[1][0]*100,2)
            
            avg_return = prior_30["pct_change"].mean()
            median_return = prior_30["pct_change"].median()
            return_std = prior_30["pct_change"].std()
            
            date_data = ticker_data[ticker_data.index ==  pd.to_datetime(date).strftime("%Y-%m-%d")].copy()
            
            regime = date_data["regime"].iloc[0]
            
            last_return = prior_30["pct_change"].iloc[-1]
            return_before_last = prior_30["pct_change"].iloc[-2]
            t_minus_3 = prior_30["pct_change"].iloc[-3]
            
            actual_return = date_data["pct_change"].iloc[0]
            
            return_dataframe = pd.DataFrame([{"date": date,"ticker": ticker, "correlation": historical_correlation, "avg_return": avg_return,
                                              "median_return": median_return, "return_before_last": return_before_last, "last_return": last_return,
                                              "t-3": t_minus_3,"return_std": return_std, "regime": regime, "actual_return": actual_return}])
            
            ticker_data_list.append(return_dataframe)
            
        full_ticker_data = pd.concat(ticker_data_list).reset_index(drop=True).set_index("date")
        
        full_ticker_data["actual_return"] = full_ticker_data["actual_return"].shift(-1)
        full_ticker_data = full_ticker_data.dropna()
        
        full_ticker_data_list.append(full_ticker_data)
    
        complete_ohlcv_data = pd.concat(complete_ohlcv_list)    
        complete_ticker_data = pd.concat(full_ticker_data_list)
            
        # =============================================================================
        # Backtesting
        # =============================================================================
        
        backtest_dates = calendar.schedule(start_date = "2023-01-01", end_date = today).index.strftime("%Y-%m-%d").values
        
        prediction_list = []
        times = []
        
        for trade_date in backtest_dates:
            
            day_ticker_data = complete_ticker_data[complete_ticker_data.index.date == pd.to_datetime(trade_date).date()].copy()
            
            if len(day_ticker_data) < 1:
                continue
            
            daily_tickers = day_ticker_data["ticker"].values
            
            for eligible_ticker in daily_tickers:
            
                try:
                    
                    polygon_date = trade_date
                    
                    elgiible_ticker_data = complete_ticker_data[complete_ticker_data["ticker"] == eligible_ticker].copy()
                    elgiible_ticker_ohlcv_data = complete_ohlcv_data[complete_ohlcv_data["ticker"] == eligible_ticker].copy()
                    
                    historical_data = elgiible_ticker_data[elgiible_ticker_data.index.date < pd.to_datetime(trade_date).date()].copy().tail(504)
                
                    X = historical_data.drop(["actual_return", "correlation", "ticker"], axis = 1)
                    Y = historical_data["actual_return"].apply(binarizer)
                    
                    GradientBoosing_Model = GradientBoostingClassifier(n_estimators = 200).fit(X,Y)
                    
                    oos_data = elgiible_ticker_data[elgiible_ticker_data.index.date == pd.to_datetime(trade_date).date()].copy()
                    oos_price_data = elgiible_ticker_ohlcv_data[elgiible_ticker_ohlcv_data.index.date >= pd.to_datetime(trade_date).date()].copy()
                
                    X_test = oos_data.drop(["actual_return", "correlation", "ticker"], axis = 1)
                    
                    y_pred = GradientBoosing_Model.predict(X_test)
                    y_pred_proba = GradientBoosing_Model.predict_proba(X_test)
                    y_pred_proba = [y_pred_proba[0][y_pred[0]]]
                    y_test = oos_data["actual_return"].apply(binarizer).iloc[0]
                    
                    price = oos_price_data["c"].iloc[0]
                    next_open_price = oos_price_data["o"].iloc[1]
                    next_close_price = oos_price_data["c"].iloc[1]
                    
                    if y_pred[0] == 1:
                        
                        open_pnl = next_open_price - price
                        close_pnl = next_close_price - price
                        
                    elif y_pred[0] == 0:
            
                        open_pnl = price - next_open_price
                        close_pnl = price - next_close_price
                        
                    open_pnl_percent = round((open_pnl / price) * 100 , 2)
                    close_pnl_percent = round((close_pnl / price) * 100 , 2)
                    
                    pred_df = pd.DataFrame({"pred": y_pred, "proba":y_pred_proba, "actual": y_test,
                                            "return_std": oos_data["return_std"].iloc[0],"price": price,
                                            "open_pnl": open_pnl, "open_pnl_percent": open_pnl_percent, "close_pnl": close_pnl,
                                            "close_pnl_percent": close_pnl_percent,
                                            "date": oos_data.index[0], "ticker": eligible_ticker})
                    
                    prediction_list.append(pred_df)
                    
                except Exception as error:
                    print(error, eligible_ticker)
                    continue
                
        full_prediction_data = pd.concat(prediction_list)

        accuracy_rate = round(len(full_prediction_data[full_prediction_data["pred"] == full_prediction_data["actual"]]) / len(full_prediction_data),2)
        
        prediction_stats = pd.DataFrame([{"ticker": eligible_ticker, "accuracy_rate": accuracy_rate}])
        full_prediction_list.append(prediction_stats)
        print(f"Accuracy Rate: {accuracy_rate*100}%, {ticker}")
        
    except Exception as meta_error:
        print(meta_error)
        continue
    
all_predictions = pd.concat(full_prediction_list)

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
all_predictions.to_sql("prediction_accuracy", con=engine, if_exists = "replace")
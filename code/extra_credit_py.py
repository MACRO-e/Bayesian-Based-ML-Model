# -*- coding: utf-8 -*-
"""
Created on Sat May 27 19:55:29 2023

@author: 17795
"""


import pandas as pd 
import numpy as np
import riskfolio as rp
import pyfolio


def get_pnl(test_data: pd.DataFrame) -> pd.Series:
    '''
    calculate the daily pnl from weighted return data

    Returns
    -------
    pd.Series.
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
    '''
    # calculate the daily weight
    daily_weight_sum = test_data.groupby('date')['Weight'].sum().rename('daily_weight_sum')
    
    # merge daily_weight_sum into original DataFrame
    test_data = test_data.join(daily_weight_sum, on='date')
    
    # calculate weighted return
    test_data['weighted ret'] = test_data['Weight'] / test_data['daily_weight_sum'] * test_data['ret']
    
    # calculate the daily pnl
    daily_pnl = test_data[['weighted ret','date']].groupby('date').sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    
    return pd.Series(daily_pnl['weighted ret'])


def get_position(test_data: pd.DataFrame) -> pd.DataFrame:
    
    '''
    calculate the position data from weighted return data

    Returns
    -------
    positions : pd.DataFrame, optional
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375
    '''
    # calculate the daily weight
    daily_weight_sum = test_data.groupby('date')['Weight'].sum().rename('daily_weight_sum')
    
    # merge daily_weight_sum into original DataFrame
    test_data = test_data.join(daily_weight_sum, on='date')
    
    # calculate weighted return
    test_data['adj_weight'] = test_data['Weight'] / test_data['daily_weight_sum']
    
    position = test_data.pivot_table(index=['date'], columns=['id'], values=['adj_weight']).fillna(0)
    
    # generate cash 
    position['cash'] = 1 - position.sum(axis=1)
    
    return position
    


if __name__ == "__main__":
    test_data = pd.read_csv('C:/Users/17795/Documents/WeChat Files/wxid_8i8msci86byh22/FileStorage/File/2023-05/plot_data.csv')

    daily_ret = get_pnl(test_data)
    daily_ret.head()
    
    position = get_position(test_data)
    position.head()

    pyfolio.create_full_tear_sheet(returns=daily_ret,positions=position,slippage=0)
    

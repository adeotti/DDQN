from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

import sys
from dataclasses import dataclass

import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.width",None)


@dataclass(frozen=True)
class env_hypers_params():
    start_date = "1990-01-01"
    end_date = "2026-05-05"
    
    ticker_list = [ # XIU + TSX heavy components (marketcap > 10B) + Gold 
            "RY.TO", "TD.TO", "SHOP.TO", "BMO.TO", "ENB.TO", "CM.TO",
           "BNS.TO", "AEM.TO", "CNQ.TO", "SU.TO", "XIU.TO","XGD.TO"
    ]  
    indicators = [
            "macd","rsi_14","volume_30_sma","volume_100_sma","dx_30",
            "close_30_sma","close_100_sma","boll_ub","boll_lb","atr_14" 
    ]

    hmax = 10_000
    init_amount = 10_000 # initial amount
    state_dim = 145
    reward_scale = 1e-2
    

hypers = env_hypers_params()

def __process_data():
    x = YahooDownloader(hypers.start_date,hypers.end_date,hypers.ticker_list)
    x = x.fetch_data()
    fe = FeatureEngineer(True,hypers.indicators,use_vix=True) 
    info = fe.preprocess_data(x)
    info.fillna(0,inplace=True)
    #-
    info = info.sort_values(["date", "tic"])
    info = info.reset_index(drop=True)
    info.index = info.date.factorize()[0]
    # -
    total_row = len(info)
    train_len = int(total_row * 0.8)  # train/test ratio = 80/20
    train_data = info.iloc[:train_len]
    test_data = info.iloc[train_len:]
    assert len(train_data) + len(test_data) == len(info)
    return train_data,test_data

def __build_train_env():
    train_data,_ = __process_data()

    x = StockTradingEnv(
        df = train_data,
        hmax = hypers.hmax,
        initial_amount = hypers.init_amount,
        state_space = hypers.state_dim,
        action_space = 12,
        reward_scaling = hypers.reward_scale,
        num_stock_shares = [0]*12,
        buy_cost_pct = [0.001]*12,
        sell_cost_pct = [0.001]*12,
        stock_dim = 12,
        tech_indicator_list = hypers.indicators,
    ) 
    return x
    

if __name__ == "__main__":
    env = __build_train_env() 
    print(len(env.reset()[0]))
    print(env.action_space)
    

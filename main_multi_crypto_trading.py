"""Training and Backtesting Crypto Trading Bot with
   Binance Historycal Data
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import json
import itertools
from opts import parse_opts
import hyperopt_params

import ray
import optuna

from finrl.marketdata.binance_data import BinanceData_dl
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv as gym_environment
from finrl.env.env_dydx3 import dydxTradingEnv as gym_environment

# from finrl.env.crypto_env import CryptoEnv as gym_environment
import finrl.env.crypto_env as crypto_env

from finrl.model.models import DRLAgent, DRLEnsembleAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.config.config_crypt import BINANCE_TICKER

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

opt = parse_opts()
print(opt)

class LoggingCallback:
    def __init__(self,threshold,trial_number,patience):
      '''
      threshold:int tolerance for increase in sharpe ratio
      trial_number: int Prune after minimum number of trials
      patience: int patience for the threshold
      '''
      self.threshold = threshold
      self.trial_number  = trial_number
      self.patience = patience
      self.cb_list = [] #Trials list for which threshold is reached
    def __call__(self,study:optuna.study, frozen_trial:optuna.Trial):
      #Setting the best value in the current trial
      study.set_user_attr("previous_best_value", study.best_value)
      
      #Checking if the minimum number of trials have pass
      if frozen_trial.number >self.trial_number:
          previous_best_value = study.user_attrs.get("previous_best_value",None)
          #Checking if the previous and current objective values have the same sign
          if previous_best_value * study.best_value >=0:
              #Checking for the threshold condition
              if abs(previous_best_value-study.best_value) < self.threshold: 
                  self.cb_list.append(frozen_trial.number)
                  #If threshold is achieved for the patience amount of time
                  if len(self.cb_list)>self.patience:
                      print('The study stops now...')
                      print('With number',frozen_trial.number ,'and value ',frozen_trial.value)
                      print('The previous and current best values are {} and {} respectively'
                              .format(previous_best_value, study.best_value))
                      study.stop()

def get_data(opt, fname_raw: str, fname_processed: str):
    # Run prep routines
    opt = parse_opts()
    
    if opt.tickers == '*' or not opt.tickers:
        tickers = BINANCE_TICKER
    else:
        tickers = opt.tickers
        
    if opt.time_resolution not in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"]:
        print("Invalid time resolution. Please choose from 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")
        exit()
    
    # Download Data
    print("start date:",opt.start_date)
    print("end date:",opt.end_date)
    print("data_dir:",opt.data_dir)
    print("list of tickers:",opt.tickers)

    if opt.get_data:
        if opt.market == "Binance":
            df = BinanceData_dl(start_date = opt.start_date,
                                end_date = opt.end_date,
                                data_dir = opt.data_dir,
                                ticker_list = tickers,
                                time_resolution = opt.time_resolution).fetch_data()
        
        df.to_csv(fname_raw)
    else:
        df = pd.read_csv(fname_raw)
    


    # Preprocess Data
    return df


def add_features(df, indicators: list):
    print("Feature Engineering:")
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list = indicators,
                         #use_turbulence=True,
                         use_turbulence=False,
                         user_defined_feature = False)
    processed = fe.preprocess_data(df)

    # save processed data
    # processed.to_csv(fname_processed)

    processed_full = processed
    processed_full = processed_full.sort_values(['date','tic']).reset_index(drop=True)
    
    return processed_full

def calculate_sharpe(df, risk_free_rate=0):
    df["returns"] = df["account_value"].pct_change()

    # Calculate the average daily return and standard deviation of daily returns
    avg_return = df["returns"].mean()
    std_return = df["returns"].std()

    # Calculate the annualized average return and standard deviation
    avg_annual_return = (1 + avg_return) ** 252 - 1
    std_annual_return = std_return * np.sqrt(252)

    # Calculate the Sharpe ratio
    sharpe_ratio = (avg_annual_return - risk_free_rate) / std_annual_return

    return sharpe_ratio


def objective(trial:optuna.Trial):
    #Trial will suggest a set of hyperparamters from the specified range
    if opt.tune_indicators:
        tech_indicators = []
        i = 0
        tech_indicators = ['macd', 'boll_ub', 'boll_lb']
        for _ in range(0,5):
            tech_indicators.append(hyperopt_params.sma(trial, i))
            i += 1
        for _ in range(0,2):
            tech_indicators.append(hyperopt_params.rsi(trial, i))
            i +=1
        tech_indicators = set(tech_indicators)
    else: 
        tech_indicators = ['macd', 'boll_ub', 'boll_lb', 'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_50_sma', 'close_100_sma', 'close_200_sma', 'rsi_14']
    print(tech_indicators)
    
    
    
    df = get_data(opt, fname_raw, fname_processed)            
    processed_full = add_features(df, tech_indicators)
    
    # print(processed_full.head)
    
    print(f'full data length: {len(processed_full)}')
    
    train = data_split(processed_full, opt.start_date, opt.start_trade_date)
    trade = data_split(processed_full, opt.start_trade_date, opt.end_date)
    
    train = train.drop(['Unnamed: 0','date','tic'], axis=1)
    trade = trade.drop(['Unnamed: 0','date','tic'], axis=1)
    
    
    # stock_dimension = len(train.tic.unique())
    # state_space = 1 + 2*stock_dimension + len(tech_indicators)*stock_dimension
    # print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    trials_dir = f"./trials/{opt.model_name}/{trial.number}"
    if not os.path.exists(trials_dir):
        os.makedirs(trials_dir)
    
    # kwarg_list = crypto_env.required_args()
    
    
    # env_kwargs = {
    #     "hmax": opt.hmax,
    #     "initial_amount": 10000.0,
    #     "buy_cost_pct": 0,
    #     "sell_cost_pct": 0,
    #     "state_space": state_space,
    #     "stock_dim": stock_dimension,
    #     "tech_indicator_list": tech_indicators,
    #     "action_space": stock_dimension,
    #     "reward_scaling": 1e-4,
    #     "make_plots": True
    # }
    
    # env_kwargs = {'window_size': 10, 'frame_bound': (10, len(train))}
    
    e_train_gym = gym_environment(df = train)

        # environment for training
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env_train)
    hyperparameters = hyperopt_params.HYPERPARAMS_SAMPLER[opt.model_name](trial)
    model = agent.get_model(opt.model_name, model_kwargs = hyperparameters )
    # model = agent.get_model(opt.model_name )
    
    #You can increase it for better comparison
    trained_model = agent.train_model(model=model,
                                    tb_log_name=opt.model_name,
                                total_timesteps=50000)

    trained_model.save(f'{trials_dir}/model')
    e_test_gym = gym_environment(df = trade)
    final_portfolio_value = agent.evaluate(
        eval_env = e_test_gym,
        model=trained_model
        )
    
    print(f'final account_value: {final_portfolio_value}')
    
    parameters = {'tech_indicators' : list(tech_indicators),
                  'final_account_value' : final_portfolio_value}
    print(f'parameters: {parameters}')
    with open(f'{trials_dir}/paramaters.json', 'w') as f:
        f.write(json.dumps(parameters, indent=4))
    
    

    return final_portfolio_value


if __name__ == "__main__":
    # parse command-line options


    # Create Folders
    if not os.path.exists("./" + opt.results_dir):
        os.makedirs("./" + opt.results_dir)

    if not os.path.exists("./" + opt.data_dir):
        os.makedirs("./" + opt.data_dir)
        
    # Data Prep
    #fname_processed ="data/Binance/preprocessed_binance_1min_single.csv"
    fname_raw = f"data/raw_{opt.market}_{'-'.join(opt.tickers)}_{opt.time_resolution}.csv"  ## TODO adapt to use args for filename
    
    fname_processed = f"data/preprocessed_{opt.market}_{'-'.join(opt.tickers)}_{opt.time_resolution}.csv"  ## TODO adapt to use args for filename
    

    df = get_data(opt, fname_raw, fname_processed)            
    

    
    if opt.prep_data == 1:
        # run the prep
        if not opt.tech_indicators:
            tech_indicators = ['macd', 'boll_ub', 'boll_lb', 'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_50_sma', 'close_100_sma', 'close_200_sma', 'rsi_14']
        else:
            tech_indicators = opt.tech_indicators
        processed_full = add_features(df, tech_indicators)
    
    # print(processed_full.head)
    
        print(f'full data length: {len(processed_full)}')
    
    
    else:
        # load preprocessed data
        print("loading from preprocessed data")
        processed_full = pd.read_csv(fname_processed)
        
    # print(processed_full.sort_values(['date','tic'],ignore_index=True).head(10))
    # Design Environment
    print(f'full data length: {len(processed_full)}')
    train = data_split(processed_full, opt.start_date, opt.start_trade_date)
    trade = data_split(processed_full, opt.start_trade_date, opt.end_date)
    
    train = train.drop(['Unnamed: 0','date','tic'], axis=1)
    trade = trade.drop(['Unnamed: 0','date','tic'], axis=1) 

    
    
    print(f'Training data length: {len(train)}')
    print(f'Testing data length: {len(trade)}')
    
    if not(opt.no_train):
        # Training

        # print(type(env_train))
        
        sampler = optuna.samplers.TPESampler(seed=42)
        tuning_storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                            engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
                                        )

        if opt.reset_study:
            if os.path.exists(f'./trials/{opt.model_name}'):
              shutil.rmtree(f'./trials/{opt.model_name}')
            try: 
                optuna.delete_study(study_name=f"{opt.model_name}_study", storage=tuning_storage)
            except:
                pass
            study = optuna.create_study(study_name=f"{opt.model_name}_study",storage=tuning_storage, direction='maximize',
                                    sampler = sampler, pruner=optuna.pruners.HyperbandPruner())
        else:
            study = optuna.load_study(study_name=f"{opt.model_name}_study", storage=tuning_storage)

        if opt.hypertune:
            trials = 54
            logging_callback = LoggingCallback(threshold=1e-5,patience=30,trial_number=5)
            #You can increase the n_trials for a better search space scanning
            if len(study.trials) < trials:
                study.optimize(objective, 
                                n_trials=trials,
                                n_jobs=3,
                                catch=(ValueError,),
                                callbacks=[logging_callback], 
                                show_progress_bar=True)
                
            
        
            print('Hyperparameters after tuning',study.best_params)
            print(f'Best trial number: {study.best_trial.number}')
            print(f'Best Trial: {study.best_trial}')
            
            best_destination_dir = f"./best_trials/{opt.model_name}/{study.best_trial.number}"
            
            # if not os.path.exists(best_destination_dir):
            #     os.makedirs(best_destination_dir)
            if not os.path.isfile(f'{best_destination_dir}/model.zip'):
                shutil.copytree(f"./trials/{opt.model_name}/{study.best_trial.number}", best_destination_dir)
        else:
            e_train_gym = gym_environment(df = train)

        # environment for training
            model_kwargs = {'n_steps': 2048, 'ent_coef': 0.01, 'learning_rate': 0.00025, 'batch_size': 64}
            env_train, _ = e_train_gym.get_sb_env()
            agent = DRLAgent(env_train)
            model = agent.get_model(opt.model_name, model_kwargs=model_kwargs )
            # model = agent.get_model(opt.model_name )
            
            #You can increase it for better comparison
            trained_model = agent.train_model(model=model,
                                            tb_log_name=opt.model_name,
                                        total_timesteps=50000)

            trained_model.save(f'models/{opt.model_name}')
            e_test_gym = gym_environment(df = trade)
            final_portfolio_value = agent.evaluate(
                eval_env = e_test_gym,
                model=trained_model
                )
    
            print(f'final account_value: {final_portfolio_value}')
            

    else:
        # load pretrained model
        print("loading from pretrained model")
        trained_model = MODELS[opt.model_name].load(f'{opt.load_model}/model')
        with open(f'{opt.load_model}/paramaters.json', 'r') as f:
            params = json.load(f)


    test_gym = gym_environment(df = trade)
    agent = DRLAgent(test_gym)
    model = agent.get_model(opt.model_name )
    
    if opt.load_model and opt.no_train:
        trained_model = model.load(f'{opt.load_model}/model.zip')
    # else:        
    #    trained_model = model.load(f'{best_destination_dir}/model.zip')
    
    e_trade_gym = gym_environment(df = trade)

    final_portfolio_value = agent.evaluate(
        eval_env = test_gym,
        model=trained_model
    )
    print(f'final_account_value: {final_portfolio_value}')


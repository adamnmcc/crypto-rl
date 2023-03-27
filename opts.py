import argparse
from datetime import datetime

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.add_argument(
        '--prep_data',
        type=int,
        default=0,
        help='run prep or not')
    parser.add_argument(
        '--model_name',
        type=str,
        default='ppo',
        help='Name of RL models')
    parser.add_argument(
        '--load_model',
        type=str,
        help='Name of RL models')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Data directory path')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Results directory path')
    parser.add_argument(
        '--start_date',
        type=str,
        default='2019-01-01',
        help='Start date of the market dataset')
    parser.add_argument(
        '--end_date',
        default=datetime.now().strftime("%Y-%m-%d"),
        type=str,
        help='End date of the market dataset')
    parser.add_argument(
        '--start_trade_date',
        type=str,
        help='Trading (Testing) date of the market dataset')
    parser.add_argument(
        '--tech_indicators',
        type=str,
        nargs='*',
        default=None,
        help='Technical indicators to be used as the input of RL model')
    parser.add_argument(
        '--market',
        default='Binance',
        type=str,
        help='Which market data to be used')
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='*',
        help='Tickers to be used for trading')
    parser.add_argument(
        '--hmax',
        type=float,
        default=10000,
        help='Maximum trading volume per trade in USDT')
    parser.add_argument(
        '--initial_amount',
        type=float,
        default=10000,
        help='Initial amount in USDT')
    parser.add_argument(
        '--buy_cost_pct',
        type=float,
        default=0.1,
        help='Cost of buying in %')
    parser.add_argument(
        '--sell_cost_pct',
        type=float,
        default=0.1,
        help='Cost of selling in %')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning Rate for RL')
    parser.add_argument(
        '--total_timesteps',
        type=float,
        default=80000,
        help='Total Timesteps for RL')
    parser.add_argument(
        '--time_resolution',
        type=str,
        default='15m',
        help='Time resolution of the market dataset')
    parser.add_argument(
        '--get_data',
        action='store_true',
        help='If true, data is downloaded performed.')
    parser.add_argument(
        '--reset_study',
        action='store_true',
        help='If true, data is downloaded performed.')
    parser.add_argument(
        '--tune_indicators',
        action='store_true',
        help='If true, use Optuna to tune indicators.')
    parser.add_argument(
        '--hypertune',
        action='store_true',
        help='If true, use Optuna to tune gym environment parameters.')
    args = parser.parse_args()

    return args

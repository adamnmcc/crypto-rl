import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger

from .trading_env import TradingEnv, Actions, Positions




class CryptoEnv(TradingEnv):
    def __init__(self, df, window_size, frame_bound, unit_side="right"):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ["left", "right"]

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__(df, window_size)

        self.trade_fee = 0.005  # unit

    def _process_data(self):
        prices = self.df.loc[:, "close"].to_numpy()

        prices[
            self.frame_bound[0] - self.window_size
        ]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0  # pip

        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long:
                step_reward += price_diff * 10000

        return step_reward

    def _update_profit(self, action):
        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self.unit_side == "left":
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == "right":
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Short
            else:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == "left":
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == "right":
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    
def required_args():
    return ["df", "window_size", "frame_bound"]
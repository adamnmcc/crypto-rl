import numpy as np
# import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class dydxTradingEnv(gym.Env):
    def __init__(self, df, leverage=2, initial_balance=10000, transaction_cost=0.00, max_leverage=5, stop_loss_pct=0.05, take_profit_pct=0.05):
        super(dydxTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_leverage = max_leverage
        self.balance = self.initial_balance
        self.positions = 0
        self.current_step = 0
        self.leverage = 1
        self.net_worth_history = [self.initial_balance]
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.info = {}


        # Define action_space: Discrete(3) - 0: hold, 1: buy, 2: sell
        self.action_space = spaces.Discrete(4)

        # Define observation_space: Continuous Box space (price data + technical indicators + balance + positions + leverage)
        n_prices = len(self.df.columns)
        print(f'initial df length: {n_prices}')
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(n_prices + 3,))
        
        # print(self.df.head)

    def _calculate_sharpe_ratio(self):
        if len(self.net_worth_history) < 2:
            return 0.0

        returns = np.diff(self.net_worth_history)
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        sharpe_ratio = mean_returns / std_returns if std_returns != 0 else 0

        return sharpe_ratio


    def _update_leverage(self):
        sharpe_ratio = self._calculate_sharpe_ratio()
        self.leverage = max(1, min(self.max_leverage, int(sharpe_ratio)))

    def reset(self):
        self.balance = self.initial_balance
        self.positions = 0
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        # Get the current stock price
        current_price = self.df.loc[self.current_step, "close"]

        # Update the balance and positions based on the action
        if action == 1:  # Buy
            max_affordable = self.balance // (current_price * (1 + self.transaction_cost))
            buy_amount = max_affordable // self.leverage
            buy_value = buy_amount * current_price
            buy_value_with_cost = buy_value * (1 + self.transaction_cost)

            if buy_value_with_cost <= self.balance:
                self.positions += buy_amount * self.leverage
                self.balance -= buy_value_with_cost
        elif action == 2:  # Sell
            sell_amount = self.positions
            sell_value = sell_amount * current_price
            sell_value_with_cost = sell_value * (1 - self.transaction_cost)

            self.positions = 0
            self.balance += sell_value_with_cost
        elif action == 3:  # Short
            max_affordable_short = self.balance // (current_price * (1 + self.transaction_cost))
            short_amount = max_affordable_short // self.leverage
            short_value = short_amount * current_price
            short_value_with_cost = short_value * (1 + self.transaction_cost)

            if short_value_with_cost <= self.balance:
                self.positions -= short_amount * self.leverage
                self.balance -= short_value_with_cost

        # Check if the stop-loss condition is met for long positions
        if self.positions > 0:
            buy_price = self.balance / (self.positions * (1 - self.transaction_cost))
            if current_price < buy_price * (1 - self.stop_loss_pct):
                sell_amount = self.positions
                sell_value = sell_amount * current_price
                sell_value_with_cost = sell_value * (1 - self.transaction_cost)

                self.positions = 0
                self.balance += sell_value_with_cost
                stop_loss_triggered = True
            else:
                stop_loss_triggered = False
        # Check if the stop-loss condition is met for short positions
        elif self.positions < 0:
            short_price = self.balance / (abs(self.positions) * (1 + self.transaction_cost))
            if current_price > short_price * (1 + self.stop_loss_pct):
                cover_amount = abs(self.positions)
                cover_value = cover_amount * current_price
                cover_value_with_cost = cover_value * (1 + self.transaction_cost)

                self.positions = 0
                self.balance -= cover_value_with_cost
                stop_loss_triggered = True
            else:
                stop_loss_triggered = False
        else:
            stop_loss_triggered = False

        # Calculate the reward as the net worth change
        net_worth_before = self.balance + self.positions * current_price
        net_worth_after = self.balance + self.positions * self.df.loc[self.current_step + 1, "close"]
        reward = net_worth_after - net_worth_before

        # Update the net worth history
        self.net_worth_history.append(float(net_worth_after))

        # Update the leverage based on risk/reward
        self._update_leverage()


        done = (self.current_step >= self.df.shape[0] - 2)
        if done:
            return self._get_observation(), reward, done, self.info
        # Increment the step counter and check if the episode is done
        self.current_step += 1
       
        

        # Update the reward if the stop-loss was triggered
        if stop_loss_triggered:
            reward -= 100

         # Calculate the portfolio value
        portfolio_value = self.balance + self.positions * self.df.loc[self.current_step + 1, "close"]
        
        # Include the portfolio value in the info dictionary
        self.info['portfolio_value'] = portfolio_value.item()
        
        
        
        return self._get_observation(), reward, done, self.info

    def _get_observation(self):
        # Get the stock price data and technical indicators for the current step
        data_with_indicators = self.df.loc[self.current_step].tolist()

        # Normalize the balance and positions for better stability during training
        normalized_balance = self.balance / self.initial_balance
        normalized_positions = self.positions / (self.initial_balance / self.df.loc[0, "close"])

        data_with_indicators_values = np.array(data_with_indicators).reshape(1, -1)

        # Normalize and reshape balance, positions, and leverage arrays
        normalized_balance = np.array([normalized_balance]).reshape(-1, 1)
        normalized_positions = np.array([normalized_positions]).reshape(-1, 1)
        self.leverage = np.array([self.leverage]).reshape(-1, 1)

        # Combine the stock price data, technical indicators, balance, positions, and leverage into an observation array
        observation = np.hstack([data_with_indicators_values, normalized_balance, normalized_positions, self.leverage])

        return observation

    # The reset and _calculate_reward functions remain the same

    def _calculate_reward(self):
        # Get the current stock price
        current_price = self.df.loc[self.current_step, "close"]

        # Calculate the net worth before and after the current step
        net_worth_before = self.balance + self.positions * current_price
        net_worth_after = self.balance + self.positions * self.df.loc[self.current_step + 1, "close"]

        # Calculate the reward as the difference in net worth
        reward = net_worth_after - net_worth_before

        return reward
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


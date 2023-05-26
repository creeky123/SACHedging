from __future__ import print_function, absolute_import
from Simulations.SimEngine import *

import numpy as np
import os

import glob

## Environment functions for hedge projection environment

## v1.0


env_dict = dict()
# general projection params
env_dict['hedge_type'] = 'static call rebalancing'  # 'static call', 'static and dynamic',
env_dict['rebalancing'] = 'daily'  # extend to weekly/once per episode
env_dict['current_timestep'] = 0
env_dict['days'] = 252

# assets and heston calibration
env_dict['mean'] = np.array([0.0, 0.0]) # stock and vol driver
env_dict['correlation'] = 0.8
env_dict['call_strikes'] = np.array([110.0, 100.0, 90.0])
env_dict['risk_free'] = 0.012   # risk free rate
env_dict['s_o'] = 100.0         # initial stock prices
env_dict['v_0'] = 0.25          # initial volatility
env_dict['theta'] = 0.25        # long run average volatility
env_dict['kappa'] = 3           # mean reversion
env_dict['xi'] = 0.3            # vol of vol
env_dict['iteration_steps'] = 1500 # discretization steps in euler scheme
env_dict['shock_percent'] = 0.05 # shock magnitude

# fia assumptions
env_dict['base_annual_lapse_prob'] = 0.04
env_dict['base_annual_death_prob'] = 0.01
env_dict['FIA_AV'] = 100
env_dict['dynamic_lapse'] = False
env_dict['switch_index'] = False
env_dict['lock_in'] = False
env_dict['partial_withdrawal'] = False

# cuda specifications
env_dict['cuda_blocks'] = 128
env_dict['cuda_threads'] = 128


## hedge projection class ##
class HedgeProjectionEnvironment:
    version = 1.0

    def __init__(self, environment_dict):
        self.env_dict = environment_dict
        self.Current_Env = None
        self.episode_length = environment_dict['days']

        self.use_calls = environment_dict['use_calls']
        self.use_puts = environment_dict['use_puts']
        self.use_stock = environment_dict['use_stock']
        self.trans_cost = environment_dict['trans_cost']

        ## Presaved_scenarios
        self.Stored_Scen_location = environment_dict['scenario_location']

        ## fia assumptions
        self.base_annual_lapse_prob = env_dict['base_annual_lapse_prob']
        self.base_annual_death_prob = env_dict['base_annual_death_prob']
        self.daily_lapse = (1 - self.base_annual_lapse_prob)**(1 / self.episode_length)
        self.daily_death = (1 - self.base_annual_death_prob)**(1 / self.episode_length)


        ## Working Env Variables ##
        self.fia = 1
        self.calls = len(environment_dict['call_strikes'])
        self.call_strikes = environment_dict['call_strikes']
        self.puts = len(environment_dict['put_strikes'])
        self.put_strikes = environment_dict['put_strikes']
        self.action_scaler = environment_dict['action_scaler']


        self.append_positions = environment_dict['include_state_positions']

        self.call_notional = np.zeros(self.calls) ## add for dynamic
        self.put_notional = np.zeros(self.puts)
        self.stock_notional = np.zeros(1)

        self.call_notional_change = np.zeros(self.calls) ## add for dynamic
        self.put_notional_change = np.zeros(self.puts)
        self.stock_notional_change = np.zeros(1)

        self.Episode_Path_Sim = np.zeros((environment_dict['days'], (self.fia*2 + self.calls*2 + self.puts*2 + 3))) # fia ov/delta, call ov/delta, s_t, v_t, T
        self.Term_State = False

        ## headers for saving sims to file
        self.headers = ['T', 's_t', 'v_t']

        self.asset_value_columns = []

        ### state headers ##
        for i in range(self.fia):
            self.headers.extend(['fia_ov', 'fia_delta'])

        for i in range(self.calls):
            call_strike = str(environment_dict['call_strikes'][i])
            self.headers.extend(['call_ov_'+call_strike, 'call_delta_'+call_strike])
            self.asset_value_columns.extend((['call_ov_'+call_strike]))

        for i in range(self.puts):
            put_strike = str(environment_dict['put_strikes'][i])
            self.headers.extend(['put_ov_'+put_strike, 'put_delta_'+put_strike])
            self.asset_value_columns.extend((['put_ov_' + put_strike]))

        ### position headers ##
        position_headers = []

        for i in range(self.calls):
            call_strike = str(environment_dict['call_strikes'][i])
            position_headers.extend(['call_strike_' + call_strike + '_position'])

        for i in range(self.puts):
            put_strike = str(environment_dict['put_strikes'][i])
            position_headers.extend(['put_strike_' + put_strike + '_position'])

        if self.use_stock:
            self.asset_value_columns.extend((['s_t']))

        position_headers.extend(['stock_notional'])

        if self.append_positions:
            self.headers.extend(position_headers)

        self.action_headers = position_headers

        ### indexers
        self.Tidx = 0
        self.sidx = 1
        self.vidx = 2
        self.fia_idx = 3
        self.fia_delta = 4

        ## state dimensions
        self.state_dim = self.fia * 2 + self.calls * 2 + self.calls + 3
        self.Time_Step = 0

        # states
        self.death_state = 0
        self.lapse_state = 0
        self.SimEng = None


    def term_state(self):
        return self.Term_State

    def time_step(self):
        return self.Time_Step

    def action_space_size(self):
        num_actions = 0
        if self.use_calls:
            num_actions += len(self.call_strikes)

        if self.use_stock:
            num_actions += 1

        return num_actions

    def reset_env(self):

        ## Simulations can be created on the fly (cuda projection) or using stored simulations
        if self.Stored_Scen_location == None:
            self.SimEng = SimEngine(self.env_dict)
        else:
            self.SimEng.next_scenario()

        ## Working Env Variables ##
        self.fia = 1
        self.calls = len(self.env_dict['call_strikes'])
        self.Episode_Path_Sim = np.zeros((self.env_dict['days'], (self.fia*2 + self.calls*2 + 3))) # fia ov/delta, call ov/delta, s_t, v_t, T
        self.Term_State = False
        self.Time_Step = 0

        ## reset notionals ##
        self.call_notional = np.zeros(self.calls)  ## add for dynamic
        self.put_notional = np.zeros(self.puts)
        self.stock_notional = np.zeros(1)

        self.call_notional_change = np.zeros(self.calls)  ## add for dynamic
        self.put_notional_change = np.zeros(self.puts)
        self.stock_notional_change = np.zeros(1)
        ###

        # states
        self.death_state = 0
        self.lapse_state = 0

        ### normalizer
        self.normalizer = 0

    def initialize_environment(self):

        ## reset notionals ##
        self.call_notional = np.zeros(self.calls)  ## add for dynamic
        self.put_notional = np.zeros(self.puts)
        self.stock_notional = np.zeros(1)

        self.call_notional_change = np.zeros(self.calls)  ## add for dynamic
        self.put_notional_change = np.zeros(self.puts)
        self.stock_notional_change = np.zeros(1)

        ## initialize stored sims or cached sim engine
        if self.Stored_Scen_location == None:
            self.SimEng = SimEngine(self.env_dict)
        else:
            if self.SimEng == None:
                self.SimEng = CachedSimEngine(self.Stored_Scen_location)

        ## pull first option values and deltas
        option_values, option_deltas, real_world_sim = self.SimEng.initialize_env()

        ## keep episode sim path in original format, normalize state returned
        self.Episode_Path_Sim[:, self.Tidx] = np.arange(0, env_dict['days'])
        self.Episode_Path_Sim[:, self.sidx:(self.vidx+1)] = real_world_sim #row column

        ## normalizer is first asset value in real world path
        self.normalizer = self.Episode_Path_Sim[self.SimEng.current_timestep, self.sidx]

        ## combine option values, option deltas into real world path
        combined = np.zeros((option_values.shape[0]*2))
        combined[::2] = option_values
        combined[1::2] = option_deltas

        self.Episode_Path_Sim[self.SimEng.current_timestep, self.fia_idx:] = combined

        state = self.prepare_state()

        return state, 0  # state reward

    def prepare_state(self):
        ### prepare state space observation ##
        state = self.Episode_Path_Sim[self.SimEng.current_timestep, :].copy()
        state[1::2] = state[1::2] / self.normalizer
        state[0] = state[0] / 251
        if self.append_positions:
            state = np.append(state, self.call_notional)
            state = np.append(state, self.put_notional)
            state = np.append(state, self.stock_notional)
        return state

    def step_env(self):
        ## step the environment function

        ## actions are processed, environment is projected/transitioned 1 step
        ## reward + new state observation are returned
        ## if it's the first timestep in the episode 0 reward and starting state observation returned
        ## rewards are always negative (objective to get to 0 reward)

        timestep = self.Time_Step

        ### if it's the first time step, initialize and return first observation
        if timestep == 0:
            state, reward = self.initialize_environment()
            self.Time_Step += 1
            self.SimEng.current_timestep = self.Time_Step
            return state.astype(np.float_), float(reward)

        ## add the value of the stock if using stock
        stock_value = self.stock_notional * self.Episode_Path_Sim[self.SimEng.current_timestep, self.sidx] / self.normalizer

        ## if terminal step calculate and return payoff as reward
        if self.SimEng.current_timestep == env_dict['days']-1:
            self.Term_State = True
            state = self.prepare_state()
            ## asset payoff
            call_payoff = 0
            for idx in range(self.calls):
                call_payoff += (max(self.Episode_Path_Sim[timestep, self.sidx] - self.call_strikes[idx], 0) / self.normalizer) * self.call_notional[idx]

            ## liability payoff
            fia_payoff = max(self.Episode_Path_Sim[timestep, self.sidx] - self.normalizer, 0) / self.normalizer

            reward = -(abs(call_payoff + stock_value - fia_payoff))

            state[self.fia_idx] = fia_payoff

            return state.astype(np.float_), float(reward)

        ## otherwise do the projection
        ## get current period option values
        option_values, option_deltas = self.SimEng.calculate_option_values()

        ## combine into episode sim path
        combined = np.zeros((option_values.shape[0] * 2))
        combined[::2] = option_values
        combined[1::2] = option_deltas

        self.Episode_Path_Sim[self.SimEng.current_timestep, self.fia_idx:] = combined

        ## calculate payoff / reward
        fia_payoff = 0

        ## lapse and death
        ## calculate the probability of lapse and death
        lapse_rv = np.random.uniform(0, 1)
        death_rv = np.random.uniform(0, 1)

        reward = 0


        ## calculate call value
        call_value = 0
        for idx in range(1, len(option_values)):
            call_value += (option_values[idx] / self.normalizer) * self.call_notional[idx-1]

            if self.trans_cost > 0:
                reward += -abs((self.call_notional_change[idx-1]) * (option_values[idx]/ self.normalizer) * self.trans_cost)

        ## calculate lapse / death transition probability
        if lapse_rv > self.daily_lapse:
            self.lapse_state = 1
            fia_payoff = 0

        if death_rv > self.daily_death:
            self.death_state = 1
            fia_payoff = 0.8 * max(self.Episode_Path_Sim[timestep, self.sidx] - self.normalizer, 0) / self.normalizer

        ## lapse or death payoff - liquidate call position
        if self.death_state == 1 or self.lapse_state == 1:
            reward += -abs(call_value + stock_value - fia_payoff)
            self.Term_State = True
        else:
            ## deterimine if using exponential reward scaling or not
            if self.env_dict['reward_scale']: #exponentially scaled reward
                reward += -abs(math.exp(abs((call_value + stock_value - (self.Episode_Path_Sim[self.SimEng.current_timestep, self.fia_idx]/ self.normalizer))))-1)
            else: # normal reward
                reward += -abs((call_value + stock_value - (self.Episode_Path_Sim[self.SimEng.current_timestep, self.fia_idx]/ self.normalizer)))

        ## prepare and return the state space observation
        state = self.prepare_state()

        self.Time_Step += 1
        self.SimEng.current_timestep += 1

        return state, float(reward)

    def set_positions(self, action_space):
        ### set action positions
        ## any action scaling is done in this function
        ## the change in asset position is tracked to calculate the transaction costs (if included)

        adjusted_action_space = action_space * self.action_scaler

        ## reset notionals ##
        self.call_notional_change = adjusted_action_space[0:self.calls] - self.call_notional  ## add for dynamic
        self.put_notional_change = adjusted_action_space[self.calls:(self.calls+self.puts)] - self.put_notional

        ## if using stock / notional
        if self.use_stock:
            self.stock_notional_change = adjusted_action_space[(self.calls+self.puts):(self.calls+self.puts+1)] - self.stock_notional


        #set stock, put and call notional
        self.call_notional = self.call_notional_change + self.call_notional
        self.put_notional = self.put_notional_change + self.put_notional

        if self.use_stock:
            self.stock_notional = self.stock_notional_change + self.stock_notional
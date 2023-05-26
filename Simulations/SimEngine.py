import glob

from Simulations.MutliProcTools import *
from Simulations.SimEngine import *
import time
import numpy as np
import os
from Simulations.CudaFunctions import *

"""
Two simulation classes:
SimEngine will generate a real world scenario and calculate embedded option values of both call and FIA along the
real world scenario.

CachedSimEngine will load pregenerated sims given a folder path and will convert and provide to the training routine
much faster! 
"""



class SimEngine:
    def __init__(self, env_dict):

        self.days = env_dict['days']

        ## assets and heston calibration
        self.mean = env_dict['mean']
        self.correlation = env_dict['correlation']
        self.cov_matrix = np.array([[1.0, self.correlation], [self.correlation, 1.0]])
        self.rf     = env_dict['risk_free'] # risk free rate
        self.theta  = env_dict['theta']     # long run average volatility
        self.kappa  = env_dict['kappa']     # mean reversion
        self.xi     = env_dict['xi']        # vol of vol
        self.iteration_steps = env_dict['iteration_steps'] # discretization steps in euler scheme

        ## asset specific info
        self.call_strikes = env_dict['call_strikes']
        self.s_0 = env_dict['s_o']  # initial stock prices
        self.v_0 = env_dict['v_0']  # initial volatility
        self.shock_percent = env_dict['shock_percent'] # shock magnitude

        self.num_of_calls = len(self.call_strikes)
        self.dynamic_hedge = env_dict

        ## fia assumptions
        self.base_annual_lapse_prob = env_dict['base_annual_lapse_prob']
        self.base_annual_death_prob = env_dict['base_annual_death_prob']
        self.daily_lapse = (1 - self.base_annual_lapse_prob)**(1 / self.days)
        self.daily_death = (1 - self.base_annual_death_prob)**(1 / self.days)
        self.FIA_AV = env_dict['FIA_AV']
        self.FIA_lock_in = env_dict['lock_in']

        ## cuda params ##
        self.cuda_blocks  =  env_dict['cuda_blocks']
        self.cuda_threads = env_dict['cuda_threads']

        #### Working variables ####
        self.dt = 1 / self.days
        self.current_timestep = 0

        ## Generated Real World Path
        self.RealWorldPath = None

        ## current state ##
        self.death_state = None
        self.lapse_state = None
        self.lock_in = None

        ## current call position
        self.call_notional = None
        self.State_Obs = None
        self.Prev_State = None

        self.end_episode = False

    def generate_real_world_path(self):
        # generates real world path that the episode will follow
        # time step
        # sqrt dt
        sqrt_dt = np.sqrt(self.dt)

        # initial price and vol of volatility
        price = self.s_0
        vol = self.v_0

        # output path
        path_output = np.zeros((self.days, 2))

        # generate normals
        correlated_norms = np.random.multivariate_normal(self.mean, self.cov_matrix, self.days) * sqrt_dt

        ##real world drift parameters are off!
        ##should be N(0.05, 0.1) for a reasonable lognormal

        drift_adjustment = np.random.normal(0.05, 0.75, self.days)
        drift_adjustment = np.exp(drift_adjustment)  # scalar multiple on risk free rate for real world evolution

        path_output[0, 0] = price
        path_output[0, 1] = vol

        # euler scheme for heston
        # heston - s_t+1 = st * exp(r_f - v(t)/2)dt + sqrt(v(t)) * dW_S_t)
        # vol - v(t+1) = v(t) + kappa (theta - v(t))*dt + xi * sqrt(v(t)) * dW_V_t

        for i in range(1, self.days):
            sqrt_vol = np.sqrt(vol)

            # stock price evolution
            price = price * (
                np.exp((self.rf * drift_adjustment[i] - 0.5 * vol) * self.dt + correlated_norms[i, 0] * sqrt_vol))

            # volatility vol evolution
            vol = max(vol + self.kappa * (self.theta - vol) * self.dt + self.xi * sqrt_vol * correlated_norms[i, 1], 0.00001)

            # write output
            path_output[i, 0] = price
            path_output[i, 1] = vol

        return path_output

    def calc_call_ov_delta (self, s_t, v_t, strike, current_day):
        ## shock for deltas
        shock_magnitude = self.shock_percent * s_t

        ## cuda parameter setup
        threads_per_block = self.cuda_threads
        blocks = 3 * self.cuda_blocks
        split = int(threads_per_block * blocks / 3)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=np.random.randint(-50000, 250000))

        ## output array
        out = np.zeros(threads_per_block * blocks, dtype=np.float32)

        T = current_day / self.days  # num days left to maturity
        iterations = np.ceil(self.iteration_steps * T) # reduce iterations when closer to maturity
        dt = T / iterations # discretization steps

        ## simulate discounted payoffs
        compute_heston_call[blocks, threads_per_block](rng_states, s_t,
                                                       v_t, self.theta,
                                                       self.kappa, self.xi, self.rf, self.correlation, dt,
                                                       strike, shock_magnitude, out, split, iterations)
        ## calculate option value and deltas
        ov_call = out[0:split].mean()
        call_up = out[split:2 * split].mean()
        call_down = out[2 * split:3 * split].mean()

        ## calculate delta using unbiased estimator
        call_delta = (call_up - call_down) / (2 * shock_magnitude)

        return ov_call, call_delta

    def calc_fia_ov_delta(self, s_t, v_t, days):

        # 5% up and down shock for delta calculation
        shock_magnitude = self.shock_percent * s_t
        threads_per_block = self.cuda_threads
        blocks = 3 * self.cuda_blocks

        split = int(threads_per_block * blocks / 3)

        ## rng provider for cuda projection
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=np.random.randint(-50000, 250000))

        out = np.zeros(threads_per_block * blocks, dtype=np.float32)

        T = days / self.days  ## num days left
        iterations = np.ceil(self.iteration_steps * T)
        dt = T / iterations

        pr_lapse = (1 - self.base_annual_lapse_prob) ** (1 / iterations)
        pr_death = (1 - self.base_annual_death_prob) ** (1 / iterations)


        compute_heston_fia[blocks, threads_per_block](rng_states, s_t,
                                                      v_t, self.theta,
                                                      self.kappa, self.xi, self.rf, self.correlation, dt,
                                                      self.RealWorldPath[0,0], shock_magnitude, out, split, iterations,
                                                      pr_death, pr_lapse, self.FIA_lock_in, T)

        ov_fia = out[0:split].mean()
        fia_up = out[split:2 * split].mean()
        fia_down = out[2 * split:3 * split].mean()

        fia_delta = (fia_up - fia_down) / (2 * shock_magnitude)

        return ov_fia, fia_delta

    def initialize_env(self):
        # real world path generate
        self.RealWorldPath = self.generate_real_world_path()
        # calculate the option values
        option_values, option_deltas = self.calculate_option_values()

        return option_values, option_deltas, self.RealWorldPath

    def calculate_option_values(self):

        ### call price and delta ###
        call_ovs = []
        call_deltas = []
        for idx in range(0,len(self.call_strikes)): # need to update with index for strike
            call_ov, call_delta = self.calc_call_ov_delta(self.RealWorldPath[self.current_timestep, 0], #S_t
                                                          self.RealWorldPath[self.current_timestep, 1], #V_t
                                                          self.call_strikes[idx], # strike
                                                          self.days - self.current_timestep)


            call_delta = min(1.0, max(call_delta, 0))

            call_ovs.append(call_ov)
            call_deltas.append(call_delta)

        ### fia ov and delta
        fia_ov, fia_delta = self.calc_fia_ov_delta(self.RealWorldPath[self.current_timestep, 0],
                                                   self.RealWorldPath[self.current_timestep, 1],
                                                   self.days - self.current_timestep)

        fia_delta = min(1.0, max(fia_delta, 0))

        option_values  = [fia_ov]
        option_values.extend(call_ovs)
        option_deltas  = [fia_delta]
        option_deltas.extend(call_deltas)


        return np.array(option_values), np.array(option_deltas)

class CachedSimEngine:
    def __init__(self, file_path):
        self.Scenario_Files = glob.glob(file_path+"/*.csv")
        self.cache = multi_proc_load_scens(self.Scenario_Files)

        ## working variables
        self.scenidx = 0
        self.current_timestep = 0
        self.parse_stored_scenario_info()

    def parse_stored_scenario_info(self):
        self.columns = list(self.cache[0].columns)
        self.call_count = 0
        self.option_value_columns = []
        self.delta_columns = []

        for column in self.columns:
            if 'call' in column:
                self.call_count += 1

            if 'ov' in column:
                self.option_value_columns.append(column)

            if 'delta' in column:
                self.delta_columns.append(column)

        self.call_count = self.call_count / 2

        self.base_columns = self.columns[1:3]

    def initialize_env(self):
        ## load up a stored scenario - MUCH FASTER
        self.Stored_Scenario = self.cache[self.scenidx]

        ## real world path
        self.real_world_path = self.Stored_Scenario[self.base_columns].to_numpy()

        ## option values
        self.option_values = self.Stored_Scenario[self.option_value_columns].to_numpy()

        ## delta values
        self.delta_values = self.Stored_Scenario[self.delta_columns].to_numpy()

        option_values = self.option_values[self.current_timestep,:]
        option_deltas = self.delta_values[self.current_timestep,:]

        return option_values, option_deltas, self.real_world_path

    def calculate_option_values(self):
        return self.option_values[self.current_timestep, :], self.delta_values[self.current_timestep, :]

    def next_scenario(self):
        #print('next scenario')
        self.scenidx += 1
        self.current_timestep = 0



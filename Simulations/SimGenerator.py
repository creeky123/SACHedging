import numpy as np
import os
import time
from multiprocessing import *
from Simulations.SimEngine import *

"""
SimGenerator class is used to pregenerate and save sims later for training.
"""

class SimGenerator:
    version = 1.0
    save_file_path = None

    def __init__(self, environment_dict):
        self.env_dict = environment_dict
        self.episode_length = environment_dict['days']

        self.Storage = []

        ## Working Env Variables ##
        self.fia = 1
        self.calls = len(environment_dict['call_strikes'])
        self.call_strikes = environment_dict['call_strikes']
        self.Episode_Path_Sim = np.zeros((environment_dict['days'], (self.fia*2 + self.calls*2 + 3))) # fia ov/delta, call ov/delta, s_t, v_t, T


        ## headers for saving sims to file
        self.headers = ['T','s_t','v_t']

        for i in range(self.fia):
            self.headers.extend(['fia_ov','fia_delta'])

        for i in range(self.calls):
            call_strike = str(environment_dict['call_strikes'][i])
            self.headers.extend(['call_ov_'+call_strike, 'call_delta_'+call_strike])

        ### indexers
        self.Tidx = 0
        self.sidx = 1
        self.vidx = 2
        self.fia_idx = 3
        self.fia_delta = 4

        ## state dimensions
        self.Time_Step = 0
        self.saved_idx = 0


    def create_save_location (self, folder_location):
        directory_name = folder_location + '/generated_sims'
        directory_name = os.getcwd() + directory_name + '/v' + str(self.version)
        try:
            os.makedirs(directory_name)
        except:
            print('Buffer dir exists, overwriting files')

        self.save_file_path = directory_name  # write with version for future iterations


    def generate_sims(self, episodes):


        for episode in range(0, episodes):
            start = time.time()
            Episode_Path_Sim = np.zeros((self.env_dict['days'], (self.fia * 2 + self.calls * 2 + 3)))  # fia ov/delta, call ov/delta, s_t, v_t, T


            SimEng = SimEngine(self.env_dict)

            for i in range(0,self.episode_length):

                if i == 0:
                    option_values, option_deltas, real_world_sim = SimEng.initialize_env()
                    ## keep episode sim path in original format, normalize state returned
                    Episode_Path_Sim[:, self.Tidx] = np.arange(0, self.env_dict['days'])
                    Episode_Path_Sim[:, self.sidx:(self.vidx + 1)] = real_world_sim  # row column
                else:
                    ## otherwise do the projection
                    option_values, option_deltas = SimEng.calculate_option_values()

                combined = np.zeros((option_values.shape[0] * 2))
                combined[::2] = option_values
                combined[1::2] = option_deltas

                Episode_Path_Sim[SimEng.current_timestep, self.fia_idx:] = combined

                SimEng.current_timestep += 1

            self.Storage.append(Episode_Path_Sim.copy())

            if episode >= 10 and episode % 100 == 0:
                self.dump_scenarios()

            print('Episode_num: '+str(episode) +' Episode_Time: '+str(time.time() - start))

    def dump_scenarios(self):
        header = ','.join(self.headers)
        for scens in self.Storage:
            np.savetxt(self.save_file_path+'/'+str(self.saved_idx)+'.csv', scens, delimiter=',', header=header)
            self.saved_idx += 1

        self.Storage = []



def ProjectSims(env_dict, save_location):
    Generator = SimGenerator(env_dict)
    Generator.create_save_location(save_location)
    Generator.generate_sims(100000)


if __name__ == '__main__':
    env_dict = dict()
    # general projection params
    env_dict['hedge_type'] = 'static call rebalancing'  # 'static call', 'static and dynamic',
    env_dict['rebalancing'] = 'daily'  # extend to weekly/once per episode
    env_dict['current_timestep'] = 0
    env_dict['days'] = 252

    # assets and heston calibration
    env_dict['mean'] = np.array([0.0, 0.0])  # stock and vol driver
    env_dict['correlation'] = -0.3
    env_dict['call_strikes'] = [100.0]
    env_dict['risk_free'] = 0.012  # risk free rate
    env_dict['s_o'] = 100.0  # initial stock prices
    env_dict['v_0'] = 0.25  # initial volatility
    env_dict['theta'] = 0.25  # long run average volatility
    env_dict['kappa'] = 3  # mean reversion
    env_dict['xi'] = 0.3  # vol of vol
    env_dict['iteration_steps'] = 2000  # discretization steps in euler scheme
    env_dict['shock_percent'] = 0.05  # shock magnitude

    # fia assumptions / used when generating on the fly scenarios
    env_dict['base_annual_lapse_prob'] = 0.04
    env_dict['base_annual_death_prob'] = 0.01
    env_dict['FIA_AV'] = 100
    env_dict['dynamic_lapse'] = False
    env_dict['switch_index'] = False
    env_dict['lock_in'] = True
    env_dict['partial_withdrawal'] = False

    # cuda specifications
    env_dict['cuda_blocks'] = 190
    env_dict['cuda_threads'] = 512
    env_dict['scenario_location'] = 'D:/Sync/RLProject/v1.0/Consolidated'


    for i in range(3):
        p = Process(target=ProjectSims, args=(env_dict, '/gen_sim_lockin_1_call' + str(i)))
        p.start()


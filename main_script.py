import torch
from EpisodeStorage import *
from SACModel.SoftActorCritic import *
from SACModel.EnvReplayBuffer import *
from HedgeProjectionEnv import *
from Utils.Utils import *
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def train_model(run_id, env_dict, buffer_dict, model_dict):

    ## main training loop ##

    torch_device = model_dict['device']
    tracking_location = 'RunTracking'

    tracking_path = os.path.join(tracking_location, run_id, run_id) #tensorboard freakouts
    model_path = tracking_path
    # setup folder to save tracking details
    save_dictionary(env_dict, tracking_path, 'env_dict.csv')
    save_dictionary(model_dict, tracking_path, 'model_dict.csv')
    save_dictionary(buffer_dict, tracking_path, 'buffer_dict.csv')
    summary_writer = SummaryWriter(tracking_path)


    ## total number of steps
    total_steps = 0
    eval_episodes = model_dict['eval_episodes'] + 1
    total_episodes = model_dict['episodes']
    batch_size = model_dict['batch_size']



    # create environment, buffer and models
    environment = HedgeProjectionEnvironment(env_dict)
    state_space_shape = environment.initialize_environment()[0].shape[0]
    action_space_size = environment.action_space_size()
    buffer = EnvReplayBuffer(buffer_dict=buffer_dict, env_dict=env_dict, torch_device=torch_device,
                             state_space_dim=state_space_shape, action_space_dim=environment.action_space_size())

    ## create SAC model
    SACModel = SoftActorCritic(model_dict=model_dict, state_space_size=state_space_shape,
                               action_space_size=environment.action_space_size())
    SACModel.to(torch_device)
    SACModel.Actor.train()
    SACModel.Critic.train()
    SACModel.TargetCritic.train()

    ## evaluation storage class
    ep_storage = EpisodeStorage(tracking_path, state_space_size=state_space_shape, action_space_size=action_space_size,
                                env_dict=env_dict,
                                state_space_headers=environment.headers,
                                action_headers=environment.action_headers,
                                asset_value_columns=environment.asset_value_columns,
                                use_stock=env_dict['use_stock'],
                                current_position=env_dict['include_state_positions'])


    lowest_eval = -99999
    no_improve = 0  # early stopping

    ### Training loop ####
    for i in range(total_episodes):

        #### Episode start ####
        done = False
        state, reward = environment.step_env()

        #### Episode loop ####
        while not done:
            if total_steps > model_dict['random_action_steps']:
                action, log_probability = SACModel.Actor.choose_stochastic_action(
                    state_space=torch.tensor(np.array([state]), device='cuda:0').float(), training=True)
                action = action.detach().cpu().numpy()[0]
            else:
                action = np.array(np.random.uniform(-1, 1, action_space_size))

            ## set action
            environment.set_positions(action)

            ## step environment
            next_state, reward = environment.step_env()
            done = environment.term_state()

            ## add transition to buffer
            buffer.append_transition(state, action, reward, next_state, done)

            if total_steps > batch_size and total_steps % model_dict['update_interval'] == 0:
                SACModel.train_one_step(batch=buffer.sample_batch(sample_size=batch_size),
                                        writer=summary_writer, n_steps=total_steps)
            report = total_steps % 50000 == 0
            if report:
                print('Episode ' + str(i) + ' Timesteps ' + str(total_steps))
            total_steps += 1

        environment.reset_env()

        with torch.no_grad():
            if i > 1 and i % 400 == 0:
                current_eval = eval_episodes
                average_eval_reward = 0
                SACModel.eval()
                SACModel.Actor.eval()
                SACModel.Critic.eval()
                SACModel.TargetCritic.eval()


                for idx in range(current_eval):
                    ep_storage.init_episode_storage()

                    #### Episode start ####
                    episode_reward = 0
                    done = False
                    state, reward = environment.step_env()

                    while not done:
                        time_step = state[0] * 251

                        action = SACModel.Actor.choose_deterministic_action(torch.tensor(np.array([state]), device='cuda:0').float())
                        action = action.detach().cpu().numpy()[0]

                        ## set action
                        environment.set_positions(action)

                        ## step environment
                        next_state, reward = environment.step_env()
                        episode_reward += reward

                        ep_storage.append_state_action(state, action * env_dict['action_scaler'], reward, time_step)

                        done = environment.term_state()
                        state = next_state

                    ep_storage.append_state_action(state, action, 0.0, state[0] * 251)

                    average_eval_reward += episode_reward / eval_episodes

                    environment.reset_env()

                summary_writer.add_scalar('Model Accuracy/Average Eval Reward', average_eval_reward, i)
                print('Average eval reward: '+str(average_eval_reward))
                ep_storage.plot_episode_data(i, average_eval_reward)

                SACModel.Actor.train()
                SACModel.Critic.train()
                SACModel.TargetCritic.train()

                if average_eval_reward > lowest_eval:
                    no_improve = 0
                    best_model_path = os.path.join(tracking_path, 'SAC_model_episode_' + str(i))
                    torch.save(SACModel, best_model_path)
                else:
                    no_improve += 1
        if no_improve > 10:
            break


    ### Evaluate the best model

    SACModel = torch.load(best_model_path)
    SACModel.eval()

    current_eval = 2500
    reward_storage = np.zeros((current_eval,3)) # episode, agent reward, static reward

    env_dict['reward_scale'] = False
    environment = HedgeProjectionEnvironment(env_dict)

    with torch.no_grad():

        for idx in range(current_eval):

            environment.reset_env()
            #### Episode start ####
            episode_reward = 0
            done = False
            state, reward = environment.step_env()

            while not done:
                action = SACModel.Actor.choose_deterministic_action(
                    torch.tensor(np.array([state]), device='cuda:0').float())
                action = action.detach().cpu().numpy()[0]

                ## set action
                environment.set_positions(action)

                ## step environment
                next_state, reward = environment.step_env()
                episode_reward += reward

                done = environment.term_state()
                state = next_state

            reward_storage[idx,0] = idx
            reward_storage[idx,1] = episode_reward


            environment.reset_env()

            #### Episode start Benchmark ####
            benchmark_reward = 0.0
            reward = 0.0
            done = False
            state, reward = environment.step_env()
            static_action = np.zeros(action_space_size)
            static_action[0] = 0.95 / env_dict['action_scaler']
            while not done:
                time_step = state[0] * 251

                ## set action
                environment.set_positions(static_action)

                ## step environment
                next_state, reward = environment.step_env()
                benchmark_reward  += reward

                done = environment.term_state()
                state = next_state

            reward_storage[idx, 2] = benchmark_reward

            if idx % 1000 == 0:
                print(idx)

    ### Save best model results ###

    df = pd.DataFrame(reward_storage)
    df.columns = ['TimeStep', 'Agent_Reward', 'Benchmark_Reward']
    df.to_csv(os.path.join(tracking_path,'BestModelEvalResults.csv'))

if __name__ == '__main__':


    ## create environment dictionary

    env_dict = dict()
    # general projection params
    env_dict['hedge_type'] = 'static call rebalancing'  # 'static call', 'static and dynamic',
    env_dict['rebalancing'] = 'daily'  # extend to weekly/once per episode
    env_dict['current_timestep'] = 0
    env_dict['days'] = 252

    # assets and heston calibration
    env_dict['mean'] = np.array([0.0, 0.0])  # stock and vol driver
    env_dict['correlation'] = 0.8
    env_dict['call_strikes'] = [100.0]
    env_dict['risk_free'] = 0.012  # risk free rate
    env_dict['s_o'] = 100.0  # initial stock prices
    env_dict['v_0'] = 0.25  # initial volatility
    env_dict['theta'] = 0.25  # long run average volatility
    env_dict['kappa'] = 3  # mean reversion
    env_dict['xi'] = 0.3  # vol of vol
    env_dict['iteration_steps'] = 1500  # discretization steps in euler scheme
    env_dict['shock_percent'] = 0.05  # shock magnitude

    # fia assumptions / used when generating on the fly scenarios
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
    env_dict['scenario_location'] = None# 'D:/Sync/RLProject/v1.0/Consolidated'
    env_dict['action_scaler'] = 1.0

    env_dict['put_strikes'] = []
    env_dict['use_calls'] = True
    env_dict['use_stock'] = True
    env_dict['use_puts'] = True
    env_dict['trans_cost'] = 0
    env_dict['include_state_positions'] = True



    # buffer specifications
    buffer_dict = dict()
    buffer_dict['seed'] = 1337
    buffer_dict['max_buff_size'] = 5000000
    buffer_dict['save_location'] = '/Buffer'

    # model specifications
    model_dict = dict()
    model_dict['random_action_steps'] = 500000 ## steps to take random action

    model_dict['learning_rate'] = 0.0001
    model_dict['tau'] = 0.005
    model_dict['gamma'] = 0.99
    model_dict['alpha'] = 0.2
    model_dict['alpha_tune'] = True

    model_dict['batch_size'] = 1024
    model_dict['episodes'] = 10000
    model_dict['update_interval'] = 25
    model_dict['eval_episodes'] = 160

    env_dict['use_stock'] = False
    env_dict['trans_cost'] = 0
    model_dict['action_scaler'] = 1.0
    model_dict['alpha'] = 0.2
    env_dict['include_state_positions'] = False
    model_dict['device'] = 'cuda:0'
    env_dict['reward_scale']=False

    processes = []

    env_dict['include_state_positions'] = False
    env_dict['use_stock'] = False
    env_dict['trans_cost'] = 0.00
    env_dict['action_scaler'] = 2.0
    model_dict['alpha'] = 0.2
    model_dict['batch_size'] = 256
    env_dict['reward_scale'] = False
    p = Process(target=train_model,
                args=('256_02alpha_2scale_03lr_0trans_nostock_noreward', env_dict, buffer_dict, model_dict))
    p.start()
    processes.append(p)


    env_dict['include_state_positions'] = False
    env_dict['use_stock'] = False
    env_dict['trans_cost'] = 0.00
    env_dict['action_scaler'] = 2.0
    model_dict['alpha'] = 0.02
    model_dict['batch_size'] = 1024
    env_dict['reward_scale'] = False
    p = Process(target=train_model,
                args=('1024_002alpha_1scale_03lr_0trans_nostock_noreward', env_dict, buffer_dict, model_dict))
    p.start()
    processes.append(p)


    env_dict['include_state_positions'] = False
    env_dict['use_stock'] = False
    env_dict['trans_cost'] = 0.00
    env_dict['action_scaler'] = 2.0
    env_dict['reward_scale'] = False
    model_dict['alpha'] = 0.2
    model_dict['batch_size'] = 1024
    env_dict['reward_scale'] = False
    p = Process(target=train_model,
                args=('1024_02alpha_2scale_03lr_0trans_nostock_noreward', env_dict, buffer_dict, model_dict))
    p.start()
    processes.append(p)
    #
    #
    #
    # env_dict['include_state_positions'] = False
    # env_dict['use_stock'] = False
    # env_dict['trans_cost'] = 0.05
    # env_dict['action_scaler'] = 2.0
    # model_dict['alpha'] = 0.02
    # model_dict['batch_size'] = 1024
    # env_dict['reward_scale'] = False
    # p = Process(target=train_model,
    #             args=('1024_002alpha_2scale_03lr_005trans_nostock_noreward', env_dict, buffer_dict, model_dict))
    # p.start()
    # processes.append(p)
    #
    #
    #
    # env_dict['include_state_positions'] = False
    # env_dict['use_stock'] = False
    # env_dict['trans_cost'] = 0.05
    # env_dict['action_scaler'] = 1.0
    # model_dict['alpha'] = 0.02
    # model_dict['batch_size'] = 1024
    # env_dict['reward_scale'] = True
    # p = Process(target=train_model,
    #             args=('1024_002alpha_1scale_03lr_05trans_nostock_reward', env_dict, buffer_dict, model_dict))
    # p.start()
    # processes.append(p)



    # env_dict['include_state_positions'] = False
    # env_dict['use_stock'] = True
    # env_dict['trans_cost'] = 0.05
    # env_dict['action_scaler'] = 2.0
    # model_dict['alpha'] = 0.02
    # model_dict['batch_size'] = 1024
    # env_dict['reward_scale'] = False
    # p = Process(target=train_model,
    #             args=('1024_002alpha_2scale_01lr_05trans_withstock_no_reward', env_dict, buffer_dict, model_dict))
    # p.start()
    # processes.append(p)
    #
    # env_dict['include_state_positions'] = True
    # env_dict['use_stock'] = True
    # env_dict['trans_cost'] = 0.05
    # env_dict['action_scaler'] = 2.0
    # model_dict['alpha'] = 0.02
    # model_dict['batch_size'] = 1024
    # env_dict['reward_scale'] = False
    # p = Process(target=train_model,
    #             args=('1024_002alpha_2scale_01lr_05trans_withstock_no_reward_with_state', env_dict, buffer_dict, model_dict))
    # p.start()
    # processes.append(p)


    # env_dict['include_state_positions'] = True
    # env_dict['use_stock'] = True
    # env_dict['trans_cost'] = 0.05
    # env_dict['action_scaler'] = 1.0
    # model_dict['alpha'] = 0.2
    # model_dict['batch_size'] = 1024
    # env_dict['reward_scale'] = True
    # p = Process(target=train_model,
    #             args=('1024_002alpha_1scale_01lr_05trans_withstock_reward_with_state', env_dict, buffer_dict, model_dict))
    # p.start()
    # processes.append(p)
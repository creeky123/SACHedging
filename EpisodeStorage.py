import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


#### class to store and graph evaluation scenarios ###
#### creates and stores graphs as it trains ###

class EpisodeStorage:
    def __init__(self, storage_dir, state_space_size, action_space_size, env_dict, state_space_headers, action_headers, asset_value_columns, use_stock=True,
                 current_position=True):
        self.episode_storage = []
        self.num_episodes = 0
        self.directory_name = os.path.join(os.getcwd(), storage_dir, 'eval_graphs')
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.proj_days = env_dict['days']
        self.current_episode = None
        self.plot_idx = 0

        if use_stock:
            self.action_columns = ['current_' + x for x in action_headers]
        else:
            self.action_columns = ['current_' + x for x in action_headers if 'stock' not in x]

        self.state_columns = state_space_headers
        self.columns = state_space_headers

        self.columns.extend(self.action_columns)
        self.columns.extend(['Reward'])
        self.asset_value_cols = asset_value_columns
        self.asset_delta_cols = [x for x in state_space_headers if 'delta' in x and 'fia' not in x]
        self.asset_delta_cols.extend(['stock_delta'])

        try:
            os.makedirs(self.directory_name)
        except:
            print('Episode storage folder exists overwriting')

    def init_episode_storage(self):
        self.reset_episode_storage()

    def reset_episode_storage(self):
        try:
            self.episode_storage.append(self.current_episode.copy())
        except:
            print('')

        self.current_episode = np.zeros((self.proj_days, self.state_space_size+self.action_space_size+1))

    def append_state_action(self, state, action, reward, timestep):
        self.current_episode[int(timestep), :] = np.concatenate((state, action, np.array([reward]))).copy()

    def calculate_agent_position(self, df):
        agent_position = np.zeros_like(df['s_t'])

        for each in zip(self.asset_value_cols, self.action_columns):
            val_col = each[0]
            pos_col = each[1]
            pos_np = df[pos_col].to_numpy()
            val_np = df[val_col].to_numpy()
            agent_position += pos_np * val_np

        return agent_position

    def calculate_agent_delta(self, df):
        agent_position = np.zeros_like(df['s_t'])
        df['stock_delta'] = np.ones_like(df['s_t'])

        for each in zip(self.asset_delta_cols, self.action_columns):
            delta_col = each[0]
            pos_col = each[1]
            pos_np = df[pos_col].to_numpy()
            delta_np = df[delta_col].to_numpy()
            agent_position += pos_np * delta_np

        return agent_position

    def plot_episode_data(self, time_steps, reward):
        output_dir = os.path.join(self.directory_name, 'Plot_Episode'+str(time_steps)+'_'+str(round(reward,1))+'_'+str(self.plot_idx))

        time_col = np.arange(0, 252)/251.0

        try:
            os.makedirs(output_dir)
        except:
            print('Episode plot folder exists overwriting')

        num_row = 4
        num_col = 3
        fig_sub1 = make_subplots(rows=num_row, cols=num_col)

        eval_num = 6
        ## select 4 best and 4 worst:
        temp_list = []
        for idx in range(0,len(self.episode_storage)):
            current_df = pd.DataFrame(self.episode_storage[idx], columns=self.columns)
            current_df['T'] = time_col
            reward = current_df['Reward'].cumsum().to_numpy()[-1]
            temp_list.append((reward, current_df))
        temp_list.sort(key=lambda x:x[0])
        plot_list = temp_list[0:eval_num]
        plot_list.extend(temp_list[-eval_num:])
        plot_list = [x[1] for x in plot_list]


        idx = 0
        for row in range(1,num_row+1):
            for col in range(1,num_col+1):
                ep_df = plot_list[idx]
                ep_df['Agent_position'] = self.calculate_agent_position(ep_df)
                ep_df = ep_df[['T', 'fia_ov', 'Agent_position']]
                ep_df.columns = ['T', 'Liability OV', 'Agent Position OV']
                ep_df = ep_df[:-1]

                for column in list(ep_df.columns[1:]):
                    if 'Agent' in column:
                        fig_sub1.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                         name=column, line=dict(color='#832DEF')), row=row,
                                              col=col)
                    else:
                        fig_sub1.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                         name=column, line=dict(color='#FF4500')), row=row,
                                              col=col)
                idx +=1

        fig_sub1.update_layout(title_text='Agent position OV vs Liability OV')
        fig_sub1.write_html(output_dir + '/' + 'fia_ov_vs_agent' + str(self.plot_idx) + ".html")
        fig_sub2 = make_subplots(rows=4, cols=4)

        idx = 0
        for row in range(1,num_row+1):
            for col in range(1,num_col+1):
                ep_df = plot_list[idx]
                ep_df['Agent_delta'] = self.calculate_agent_delta(ep_df)
                ep_df = ep_df[['T', 'fia_delta', 'Agent_delta']]
                ep_df.columns = ['T', 'Liability Delta', 'Agent Position Delta']
                ep_df = ep_df[:-1]

                for column in list(ep_df.columns[1:]):
                    if 'Agent' in column:
                        fig_sub2.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                         name=column, line=dict(color='#832DEF')), row=row,
                                             col=col)
                    else:
                        fig_sub2.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                         name=column, line=dict(color='#FF4500')), row=row,
                                              col=col)

                idx += 1

        fig_sub2.update_layout(title_text='Agent position Delta vs Liability Delta')
        fig_sub2.write_html(output_dir + '/' + 'fia_delta_vs_agent' + str(self.plot_idx) + ".html")

        fig_sub3 = make_subplots(rows=4, cols=4)

        idx = 0
        for row in range(1,num_row+1):
            for col in range(1,num_col+1):
                ep_df = plot_list[idx]
                ep_df = ep_df[['T', 's_t', 'v_t', 'Reward']]
                ep_df.columns = ['T', 'Stock Price', 'Volatility', 'Reward' ]
                ep_df = ep_df[:-1]

                for column in list(ep_df.columns[1:]):
                    if 'Stock' in column:
                        fig_sub3.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                         line=dict(color='#2e8b57'),
                                                         name=column), row=row,
                                         col=col)
                    if 'Volatility' in column:
                        fig_sub3.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                        line=dict(color='#dc143c'),
                                                        name=column), row=row,
                                             col=col)
                    if 'Reward' in column:
                        fig_sub3.append_trace(go.Scatter(x=ep_df["T"], y=ep_df[column], mode='lines',
                                                         line=dict(color='#4b0082'),
                                                         name=column), row=row,
                                              col=col)
                idx += 1

        fig_sub3.update_layout(title_text='State Space and Agent Reward')
        fig_sub3.write_html(output_dir + '/' + 'state_vs_reward' + str(self.plot_idx) + ".html")



        combined_figs = open(output_dir + '/' + 'combined_graphs_' + str(self.plot_idx) + ".html", 'w')
        combined_figs.write("<html><head></head><body>" + "\n")
        fig1_html = fig_sub1.to_html().split('<body>')[1].split('</body')[0]
        fig2_html = fig_sub2.to_html().split('<body>')[1].split('</body')[0]
        fig3_html = fig_sub3.to_html().split('<body>')[1].split('</body')[0]
        combined_figs.write(fig1_html)
        combined_figs.write(fig2_html)
        combined_figs.write(fig3_html)


        self.plot_idx += 1

        for idx, df in enumerate(plot_list):
            df.to_csv(output_dir+'/'+str(idx)+'.csv')

        self.episode_storage = []


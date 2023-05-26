import numpy as np
from numpy.random import choice, seed
import pickle
import os
import random
import torch
import time

## replay buffer for Soft Actor Critic implementation

## v1.0


## File contains Environment replay buffer class
## State, action, reward, next state transitions are stored in numpy arrays
## batches are created by randomly sampling from the buffer and passed to the mdoel code to train one iteration
## class uses a numpy array for storage, and loads the batches directly onto the the device when called

class EnvReplayBuffer:
    def __init__(self, buffer_dict, env_dict, torch_device, state_space_dim, action_space_dim):
        self.rand_gen = np.random  # random generator for sampling
        self.rand_gen.seed(buffer_dict['seed'])  # set random generator seed
        self.max_buffer_size = buffer_dict['max_buff_size']  # set the maximum buffer size
        self.environment_dict = env_dict
        self.device = torch_device
        self.state_space_size = state_space_dim
        self.action_space_size = action_space_dim

        self.current_buffer_size = 0  # to prevent repeated calls to len()
        self.state_storage = torch.zeros((self.max_buffer_size, state_space_dim), dtype=torch.float32, device=self.device)
        self.next_state_storage = torch.zeros((self.max_buffer_size, state_space_dim), dtype=torch.float32, device=self.device)
        self.reward_storage = torch.zeros((self.max_buffer_size , 1), dtype=torch.float32, device=self.device)
        self.action_storage = torch.zeros((self.max_buffer_size, self.action_space_size), dtype=torch.float32, device=self.device)
        self.term_storage = torch.zeros((self.max_buffer_size, 1), dtype=torch.float32, device=self.device)

    def append_transition(self, state, action, reward, state_next, terminal):
        # append a state obs, action, reward, terminal state transition to the buffer to be used for replay

        if self.current_buffer_size == self.max_buffer_size:
                save_idx = self.rand_gen.randint(0, self.current_buffer_size)  # randomly replace a transition in the buffer
        else:
                save_idx = self.current_buffer_size
                self.current_buffer_size += 1

        self.state_storage[save_idx,:] = torch.from_numpy(state)
        self.next_state_storage[save_idx,:] = torch.from_numpy(state_next)
        self.reward_storage[save_idx,:] = torch.from_numpy(np.array(reward))
        self.action_storage[save_idx,:] = torch.from_numpy(action)
        self.term_storage[save_idx,:] = torch.from_numpy(np.array(float(terminal)))



    def sample_batch(self, sample_size):
        ## samples a random batch and loads it straight onto the device
        ## load batch straight onto the device

        random_idx = torch.tensor(self.rand_gen.randint(0, self.current_buffer_size, size = sample_size), device=self.device)

        state_batch = torch.index_select(self.state_storage,0,random_idx)
        action_batch = torch.index_select(self.action_storage,0,random_idx)
        reward_batch = torch.index_select(self.reward_storage,0,random_idx)
        next_state_batch = torch.index_select(self.next_state_storage,0,random_idx)
        done_batch = torch.index_select(self.term_storage,0,random_idx)

        ## return the batch

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


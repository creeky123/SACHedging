Dependencies:

Numpy
Pandas
Plotly
Pytorch
Numba
(Cuda enabled GPU) 
Tensorboard

File/Folder Descriptions:

main_script - Main training & evaluation routine (trains multiple models concurrently)
HedgeProjectionEnv - Environment for reinforcement learning, takes actions, projects environment and transitions, calculates reward
EpisodeStorage - Used during training to run/store training metrics and intermediate evaluation episodes
Utils - Contains general utilities
Simulations - Contains all classes necessary to create & generate the environment instances. Cuda code for option valuation & heston implementations
            - Includes sim generator to generate and store generated simulations to disk
	        - Includes cached generator to read and stage pre-generated simulations

SACModel - Contains the model file and the replay buffer class file
Evaluation graphs - Sample of graphs produced during different training evaluation runs.


To Run:

Environment variables / options are set via the env_dict dictionary
Model variables are set via the model_dict dictionary

Training times take between 12-18 hours on a 5950x & 3090 GPU per run (30k pregenerated episodes)

30k Episodes were pretrained over the course of 3-4 days and then used for training - episodes can be generated 'on the fly'
during model training but it's incredibly compute intensive; if doing so, lower the number of block threads for cuda compute to
reduce runtime.
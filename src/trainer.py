from common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import  trange
import torch
import random
from common import util, functional
from common.functional import dict_batch_generator
import utils

class Trainer(BaseTrainer):
    def __init__(self, agent, envs_train, envs_eval, 
            env_buffer, 
            args,
            agent_batch_size=100,
            model_batch_size=256,
            rollout_batch_size=100000,
            rollout_mini_batch_size=1000,
            model_retain_epochs=1,
            num_env_steps_per_epoch=1000,
            train_model_interval=250,
            train_agent_interval=1,
            num_agent_updates_per_env_step=20, # G
            max_epoch=100000,
            max_agent_updates_per_env_step=5,
            max_model_update_epochs_to_improve=5,
            max_model_train_iterations="None",
            warmup_timesteps=5000,
            model_env_ratio=0.8,
            hold_out_ratio=0.1,
            load_path="",
            **kwargs):
        super(Trainer, self).__init__(agent, envs_train, envs_eval, **kwargs)
        self.agent = agent
        self.env_buffer = env_buffer
        self.envs_train = envs_train 
        self.envs_eval = envs_eval
        self.args = args
        #hyperparameters
        self.agent_batch_size = agent_batch_size
        self.model_batch_size = model_batch_size
        self.rollout_batch_size = rollout_batch_size
        self.rollout_mini_batch_size = rollout_mini_batch_size
        self.model_retain_epochs = model_retain_epochs
        self.num_env_steps_per_epoch = num_env_steps_per_epoch
        self.train_agent_interval = train_agent_interval
        self.train_model_interval = train_model_interval
        self.num_agent_updates_per_env_step = num_agent_updates_per_env_step
        self.max_agent_updates_per_env_step = max_agent_updates_per_env_step
        self.max_model_update_epochs_to_improve = max_model_update_epochs_to_improve
        if max_model_train_iterations == "None":
            self.max_model_train_iterations = np.inf
        else:
            self.max_model_train_iterations = max_model_train_iterations
        self.max_epoch = max_epoch
        self.warmup_timesteps = warmup_timesteps
        self.model_env_ratio = model_env_ratio
        self.hold_out_ratio = hold_out_ratio
        self.tot_env_steps = 0

        if load_path != "":
            if args.actor_type == 'smp':
                if args.morphologies[0] == 'hopper':
                    full_xml = 'environments/3d_hoppers/3d_hopper_5_full.xml'
                elif args.morphologies[0] == 'walker':
                    full_xml = 'environments/3d_walkers/3d_walker_7_full.xml'
                elif args.morphologies[0] == 'humanoid':
                    full_xml = 'environments/3d_humanoids/3d_humanoid_9_full.xml'
                elif args.morphologies[0] == 'cheetah':
                    full_xml = 'environments/3d_cheetahs/3d_cheetah_14_full.xml'
                elif args.morphologies[0] == 'whh':
                    full_xml = 'environments/3d_walkers/3d_walker_7_full.xml'
                elif args.morphologies[0] == 'cwhh':
                    full_xml = 'environments/3d_walkers/3d_walker_7_full.xml'
                print(full_xml)
                graphs = utils.getGraphStructure(
                        full_xml,
                        args.observation_graph_type,
                    )
                graph_dicts = utils.getGraphDict(
                            graphs, args.traversal_types, device=util.device
                        )
                self.agent.change_morphology(graph_dicts)
            self.load_snapshot(load_path)


    def warmup(self):
        obs_list = self.envs_train.reset()
        done_list = [False for _ in range(self.args.num_envs_train)]
        episode_timesteps_list = [0 for _ in range(self.args.num_envs_train)]
        for step in tqdm(range(self.warmup_timesteps)):
            action_list = [
                np.random.uniform(
                    low=self.envs_train.action_space.low[0],
                    high=self.envs_train.action_space.high[0],
                    size=self.args.action_max_len, 
                )
                for _ in range(self.args.num_envs_train)
            ]
            # perform action in the environment
            new_obs_list, reward_list, curr_done_list, _ = self.envs_train.step(action_list)
            reward_list = reward_list.astype(np.float32)
            curr_done_list = curr_done_list.astype(np.float32)

            for i in range(self.args.num_envs_train):
                # add the instant reward to the cumulative buffer
                # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
                done_bool = curr_done_list[i]
                if episode_timesteps_list[i] + 1 == self.args.max_episode_steps:
                    done_bool = 0
                    curr_done_list[i] = True
                
                # do not increment episode_timesteps if the sub-env has been 'done'
                if not done_list[i]:
                    episode_timesteps_list[i] += 1
                    # remove 0 padding before storing in the replay buffer (trick for vectorized env)
                    num_limbs = len(self.args.graphs[self.args.envs_train_names[i]])
                    obs = np.array(obs_list[i][: self.args.limb_obs_size * num_limbs]).astype(np.float32)
                    new_obs = np.array(new_obs_list[i][: self.args.limb_obs_size * num_limbs]).astype(np.float32)
                    action = np.array(action_list[i][:self.args.limb_action_size*num_limbs]).astype(np.float32)
                    # insert transition in the replay buffer
                    self.env_buffer[self.args.envs_train_names[i]].add_transition(
                        obs, action, new_obs, reward_list[i], done_bool
                    )

                    done_list[i] = done_list[i] or curr_done_list[i]

            obs_list = new_obs_list
            collect_done = all(done_list)

            if collect_done:

                obs_list = self.envs_train.reset()
                done_list = [False for _ in range(self.args.num_envs_train)]
                episode_timesteps_list = [0 for _ in range(self.args.num_envs_train)]




    def train(self):
        # self.post_step(self.tot_env_steps)  
        # self.save_video_demo(0, wandb_save=True)
        # return
        
        if self.tot_env_steps == 0:
            self.post_step(self.tot_env_steps)  
            self.save_video_demo(0, wandb_save=True)
            util.logger.log_str("Warming Up")
            self.warmup()
            self.tot_env_steps = self.warmup_timesteps
        
        obs_list = self.envs_train.reset()
        done_list = [False for _ in range(self.args.num_envs_train)]
        episode_reward_list = [0 for _ in range(self.args.num_envs_train)]
        episode_timesteps_list = [0 for _ in range(self.args.num_envs_train)]
        # create reward buffer to store reward for one sub-env when it is not done
        episode_reward_list_buffer = [0 for _ in range(self.args.num_envs_train)]
        
        util.logger.log_str("Started Training")

        for epoch in trange(self.max_epoch, colour='blue', desc='outer loop'): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            
            epoch_start_time = time()

            for env_step in trange(self.num_env_steps_per_epoch, colour='green', desc='inner loop'):

                self.pre_iter()
                log_infos = {}

                action_list = []
                for i in range(self.args.num_envs_train):
                    # dynamically change the graph structure of the modular policy
                    self.agent.change_morphology(self.args.graph_dicts[self.args.envs_train_names[i]])
                    # remove1 0 padding of obs before feeding into the policy (trick for vectorized env)
                    obs = np.array(
                        obs_list[i][
                            : self.args.limb_obs_size * len(self.args.graphs[self.args.envs_train_names[i]])
                        ]
                    )
                    action = self.agent.select_action(obs)
                    if self.args.expl_noise != 0:
                        action = (
                            action
                            + np.random.normal(0, self.args.expl_noise, size=action.size)
                        ).clip(
                            self.envs_train.action_space.low[0], self.envs_train.action_space.high[0]
                        )
                    # add 0-padding to ensure that size is the same for all envs
                    action = np.append(
                        action,
                        np.array([0 for _ in range(self.args.action_max_len- action.size)]),
                    )
                    action_list.append(action)


                # perform action in the environment
                new_obs_list, reward_list, curr_done_list, _ = self.envs_train.step(action_list)
                reward_list = reward_list.astype(np.float32)
                curr_done_list = curr_done_list.astype(np.float32)


                for i in range(self.args.num_envs_train):
                    # add the instant reward to the cumulative buffer
                    # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
                    episode_reward_list_buffer[i] += reward_list[i]
                    done_bool = curr_done_list[i]
                    if episode_timesteps_list[i] + 1 == self.args.max_episode_steps:
                        done_bool = 0
                        curr_done_list[i] = True
                    if curr_done_list[i] and episode_reward_list[i] == 0:
                        episode_reward_list[i] = episode_reward_list_buffer[i]
                        episode_reward_list_buffer[i] = 0
                   
                    # do not increment episode_timesteps if the sub-env has been 'done'
                    if not done_list[i]:
                        episode_timesteps_list[i] += 1
                         # remove 0 padding before storing in the replay buffer (trick for vectorized env)
                        num_limbs = len(self.args.graphs[self.args.envs_train_names[i]])
                        obs = np.array(obs_list[i][: self.args.limb_obs_size * num_limbs]).astype(np.float32)
                        new_obs = np.array(new_obs_list[i][: self.args.limb_obs_size * num_limbs]).astype(np.float32)
                        action = np.array(action_list[i][:self.args.limb_action_size*num_limbs]).astype(np.float32)
                        # insert transition in the replay buffer
                        self.env_buffer[self.args.envs_train_names[i]].add_transition(
                            obs, action, new_obs, reward_list[i], done_bool
                        )
                        self.tot_env_steps += 1

                        # record if each env has ever been 'done'
                        done_list[i] = done_list[i] or curr_done_list[i]

                
     
                obs_list = new_obs_list
                collect_done = all(done_list)

                if collect_done:

                    # #train agent
                    train_agent_start_time = time()
                    self.agent.models2train()
                    per_morph_iter = sum(episode_timesteps_list) // self.args.num_envs_train
                    for name in self.args.envs_train_names:
                        self.agent.change_morphology(self.args.graph_dicts[name])
                        agent_log_infos = {}
                        for agent_update_step in range(per_morph_iter):#*self.num_agent_updates_per_env_step):
                            agent_log_infos.update(self.train_agent(name,agent_update_step))
                            self.tot_env_steps += 1
                        log_infos.update(agent_log_infos)
                        
                    self.agent.models2eval()
                    train_agent_used_time =  time() - train_agent_start_time

                    
                    log_infos['times/train_agent'] = train_agent_used_time

                    # add to tensorboard display
                    traj_returns = []
                    traj_lengths = []
                    for i in range(self.args.num_envs_train):
                        traj_returns.append(episode_reward_list[i])
                        traj_lengths.append(episode_timesteps_list[i])
                        
                    log_infos["performance/train_return"] = np.mean(traj_returns)
                    log_infos["performance/train_length"] =  np.mean(traj_lengths)

                    obs_list = self.envs_train.reset()
                    # info_list = None
                    done_list = [False for _ in range(self.args.num_envs_train)]
                    episode_reward_list = [0 for _ in range(self.args.num_envs_train)]
                    episode_timesteps_list = [0 for _ in range(self.args.num_envs_train)]
                    # create reward buffer to store reward for one sub-env when it is not done
                    episode_reward_list_buffer = [0 for _ in range(self.args.num_envs_train)]      
                    

                self.post_step(self.tot_env_steps)  
                self.post_iter(log_infos, self.tot_env_steps)         

            epoch_end_time = time()
            epoch_duration = epoch_end_time - epoch_start_time
            util.logger.log_var("times/epoch_duration", epoch_duration, self.tot_env_steps)

            if self.tot_env_steps > self.num_env_steps_per_epoch*self.max_epoch:
                break
                  
    
    def train_agent(self, name, iter):
        train_agent_env_batch_size = self.agent_batch_size
        env_data_batch = self.env_buffer[name].sample(train_agent_env_batch_size)
        loss_dict = self.agent.update(env_data_batch, iter)
        return loss_dict
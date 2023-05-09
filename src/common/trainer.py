import numpy as np
from abc import ABC, abstractmethod
import torch
import os
import cv2
from time import time
from . import util
import imageio
import wandb
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

class BaseTrainer():
    def __init__(self, agent, train_env, eval_env, 
            max_trajectory_length,
            log_interval,
            eval_interval,
            num_eval_trajectories,
            save_video_demo_interval,
            snapshot_interval,
            **kwargs):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.max_trajectory_length = max_trajectory_length
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_trajectories = num_eval_trajectories
        self.save_video_demo_interval = save_video_demo_interval
        self.snapshot_interval = snapshot_interval
        self.last_log_timestep = 0
        self.last_eval_timestep = 0
        self.last_snapshot_timestep = 0
        self.last_video_demo_timestep = 0
        pass

    @abstractmethod
    def train(self):
        #do training 
        pass
    def pre_iter(self):
        self.ite_start_time = time()

    def post_step(self, timestep):
        log_dict = {}
        if timestep % self.eval_interval == 0 or timestep - self.last_eval_timestep > self.eval_interval:
            eval_start_time = time()
            log_dict.update(self.evaluate())
            eval_used_time = time() - eval_start_time
            avg_test_return = log_dict['performance/eval_return']
            for log_key in log_dict:
                util.logger.log_var(log_key, log_dict[log_key], timestep)
            util.logger.log_var("times/eval", eval_used_time, timestep)
            summary_str = "Timestep:{}\tEvaluation return {:02f}".format(timestep, avg_test_return)
            util.logger.log_str(summary_str)
            self.last_eval_timestep = timestep

        for loss_name in log_dict:
            util.logger.log_var(loss_name, log_dict[loss_name], timestep)




    def post_iter(self, log_dict, timestep):
        # if timestep % self.log_interval == 0 or timestep - self.last_log_timestep > self.log_interval:
        for loss_name in log_dict:
            util.logger.log_var(loss_name, log_dict[loss_name], timestep)
            # self.last_log_timestep = timestep

        if timestep % self.snapshot_interval == 0 or timestep - self.last_snapshot_timestep > self.snapshot_interval:
            self.snapshot(timestep)
            self.last_snapshot_timestep = timestep
            self.save_video_demo(timestep, wandb_save=True)
        
        if self.save_video_demo_interval > 0 and (timestep % self.save_video_demo_interval == 0 or timestep - self.last_video_demo_timestep > self.save_video_demo_interval ):
            self.save_video_demo(timestep)
            self.last_video_demo_timestep = timestep

    @torch.no_grad()
    def evaluate(self):
        traj_returns = []
        traj_lengths = []
        for traj_id in range(self.num_eval_trajectories):
            # state = self.eval_env.reset()
            obs_list = self.eval_env.reset()
            done_list = [False for _ in range(self.args.num_envs_train)]
            episode_reward_list = [0 for _ in range(self.args.num_envs_train)]
            episode_timesteps_list = [0 for _ in range(self.args.num_envs_train)]
            # create reward buffer to store reward for one sub-env when it is not done
            episode_reward_list_buffer = [0 for _ in range(self.args.num_envs_train)]
            for step in range(self.max_trajectory_length):
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
                    # add 0-padding to ensure that size is the same for all envs
                    action = np.append(
                        action,
                        np.array([0 for _ in range(self.args.action_max_len- action.size)]),
                    )
                    action_list.append(action)


                # perform action in the environment
                new_obs_list, reward_list, curr_done_list, _ = self.eval_env.step(action_list)


                for i in range(self.args.num_envs_train):
                    # add the instant reward to the cumulative buffer
                    # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
                    episode_reward_list_buffer[i] += reward_list[i]
                    if episode_timesteps_list[i] + 1 == self.args.max_episode_steps:
                        curr_done_list[i] = True
                    if curr_done_list[i] and episode_reward_list[i] == 0:
                        episode_reward_list[i] = episode_reward_list_buffer[i]
                        episode_reward_list_buffer[i] = 0
                   
                    # do not increment episode_timesteps if the sub-env has been 'done'
                    if not done_list[i]:
                        episode_timesteps_list[i] += 1
                        done_list[i] = done_list[i] or curr_done_list[i]

                # record if each env has ever been 'done'
                # done_list = [done_list[i] or curr_done_list[i] for i in range(self.args.num_envs_train)]
                        
                obs_list = new_obs_list
                collect_done = all(done_list)

                if collect_done:
                    for i in range(self.args.num_envs_train):
                        traj_lengths.append(episode_timesteps_list[i])
                        traj_returns.append(episode_reward_list[i])
                    break 

        return {
            "performance/eval_return": np.mean(traj_returns),
            "performance/eval_length": np.mean(traj_lengths)
        }
        
        
    def save_video_demo(self, ite, width=500, height=500, fps=30, wandb_save=False):

        episode_reward_list = [0 for _ in range(self.args.num_envs_train)]
        episode_timesteps_list = [0 for _ in range(self.args.num_envs_train)]
        episode_reward_list_buffer = [0 for _ in range(self.args.num_envs_train)]
        #rollout to generate pictures and write video
        obs_list = self.eval_env.reset()
        img = self.eval_env.get_images()

        video_demo_dir = os.path.join(util.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)

        imgs = []
        for i in range(img.shape[0]):
            imge = Image.fromarray(np.rot90(img[i],k=2), "RGB")
            imgs.append([imge])
        
        done_list = [False for _ in range(self.args.num_envs_train)]
        for step in range(self.max_trajectory_length):#70
            # print(step)
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
                # add 0-padding to ensure that size is the same for all envs
                action = np.append(
                    action,
                    np.array([0 for _ in range(self.args.action_max_len- action.size)]),
                )
                action_list.append(action)


            # perform action in the environment
            new_obs_list, reward_list, curr_done_list, info_list = self.eval_env.step(action_list)
            for i in range(self.args.num_envs_train):
                episode_reward_list_buffer[i] += reward_list[i]
                if episode_timesteps_list[i] + 1 == self.args.max_episode_steps:
                    curr_done_list[i] = True
                if curr_done_list[i] and episode_reward_list[i] == 0:
                    episode_reward_list[i] = episode_reward_list_buffer[i]
                    episode_reward_list_buffer[i] = 0
                # do not increment episode_timesteps if the sub-env has been 'done'
                if not done_list[i]:
                    episode_timesteps_list[i] += 1
                    done_list[i] = done_list[i] or curr_done_list[i]

            # record if each env has ever been 'done'
            # done_list = [done_list[i] or curr_done_list[i] for i in range(self.args.num_envs_train)]
                    
            obs_list = new_obs_list
            img = self.eval_env.get_images()
            
            # video_writer.write(img)
            for i in range(img.shape[0]):
                # imgs[i].append(np.rot90(img[i],k=2))
                imge = Image.fromarray(np.rot90(img[i],k=2), "RGB")
                draw = ImageDraw.Draw(imge)
                font = ImageFont.truetype("./misc/sans-serif.ttf", 20)
                draw.text(
                    (100, 10), "Distance: " + str(info_list[i]['dist']), (255, 255, 0), font=font
                )
                draw.text(
                    (100, 32), "Instant Reward: " + str(reward_list[i]), (255, 255, 0), font=font
                )
                draw.text(
                    (100, 54),
                    "Episode Reward: " + str(episode_reward_list_buffer[i]),
                    (255, 255, 0),
                    font=font,
                )
                draw.text(
                    (100, 76),
                    "Episode Timesteps: " + str(episode_timesteps_list[i]),
                    (255, 255, 0),
                    font=font,
                )
                imgs[i].append(imge)
                # if done_list[i]==False and episode_reward_list_buffer[i]!=0:
                #     img_save_path = os.path.join(video_demo_dir,str(i)+".png")
                #     imge.save(img_save_path)
            
            collect_done = all(done_list)
            if collect_done:
                break 

        for i in range(img.shape[0]):
            gif_save_path = os.path.join(video_demo_dir, str(i)+".gif")
            imageio.mimsave(gif_save_path, imgs[i], fps=60)

                


    def snapshot(self, timestamp):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, "save.pth")
        rb_path = os.path.join(save_dir, "save_")
        
        checkpoint = {
            "agent":self.agent.state_dict(),
            "tot_env_steps":self.tot_env_steps,
        }
        
        for i in range(self.args.num_envs_train): 
            checkpoint[self.args.envs_train_names[i]+"max_sample_size"]=self.env_buffer[self.args.envs_train_names[i]].max_sample_size
            checkpoint[self.args.envs_train_names[i]+"curr"]=self.env_buffer[self.args.envs_train_names[i]].curr
            np.save(
                rb_path+self.args.envs_train_names[i]+"_obs_buffer.npy",
                self.env_buffer[self.args.envs_train_names[i]].obs_buffer,
                allow_pickle=False,
            )
            np.save(
                rb_path+self.args.envs_train_names[i]+"_action_buffer.npy",
                self.env_buffer[self.args.envs_train_names[i]].action_buffer,
                allow_pickle=False,
            )
            np.save(
                rb_path+self.args.envs_train_names[i]+"_next_obs_buffer.npy",
                self.env_buffer[self.args.envs_train_names[i]].next_obs_buffer,
                allow_pickle=False,
            )
            np.save(
                rb_path+self.args.envs_train_names[i]+"_reward_buffer.npy",
                self.env_buffer[self.args.envs_train_names[i]].reward_buffer,
                allow_pickle=False,
            )
            np.save(
                rb_path+self.args.envs_train_names[i]+"_done_buffer.npy",
                self.env_buffer[self.args.envs_train_names[i]].done_buffer,
                allow_pickle=False,
            )

        torch.save(checkpoint,model_save_path)
        print("save")
        

    def load_snapshot(self, load_path):
        if not os.path.exists(load_path):
            print("\033[31mLoad path not found:{}\033[0m".format(load_path))
            exit(0)
        checkpoint = torch.load(load_path, map_location=util.device)
        self.agent.load_state_dict(checkpoint["agent"])
        print("load model",checkpoint["tot_env_steps"])
        if self.args.load_buffer:
            self.tot_env_steps = checkpoint["tot_env_steps"]
            rb_path = load_path.replace(".pth","_")
            for i in range(self.args.num_envs_train): 
                self.env_buffer[self.args.envs_train_names[i]].max_sample_size = int(checkpoint[self.args.envs_train_names[i]+"max_sample_size"])
                self.env_buffer[self.args.envs_train_names[i]].curr = int(checkpoint[self.args.envs_train_names[i]+"curr"])
                self.env_buffer[self.args.envs_train_names[i]].obs_buffer = np.load(
                    rb_path+self.args.envs_train_names[i]+"_obs_buffer.npy"
                ).astype(np.float32)
                self.env_buffer[self.args.envs_train_names[i]].action_buffer = np.load(
                    rb_path+self.args.envs_train_names[i]+"_action_buffer.npy"
                )
                self.env_buffer[self.args.envs_train_names[i]].next_obs_buffer = np.load(
                    rb_path+self.args.envs_train_names[i]+"_next_obs_buffer.npy"
                )
                self.env_buffer[self.args.envs_train_names[i]].reward_buffer = np.load(
                    rb_path+self.args.envs_train_names[i]+"_reward_buffer.npy"
                )
                self.env_buffer[self.args.envs_train_names[i]].done_buffer = np.load(
                    rb_path+self.args.envs_train_names[i]+"_done_buffer.npy"
                )
            print("load buff")

       
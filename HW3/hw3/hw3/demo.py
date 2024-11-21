import sys,os
import math
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import torch
import datetime
import time

from agents.DQN.agent import DQN
import numpy as np

import gymnasium as gym

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class DQNConfig:
    def __init__(self) -> None:
        self.algo = 'DQN'
        # self.env = "CartPole-v1"
        # self.env = "ALE/SpaceInvaders-ram-v5"
        self.env = "LunarLander-v3"
        self.seed = 15
        self.ShowImage = True     # render image
        # self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.load_model = False     # load model
        self.train = False
        # self.model_path = 'saved_models/SpaceInvaders/'  # path to save models
        self.model_path = 'saved_models/LunarLander/'  # path to save models
        self.capacity = int(2e5)    # replay buffer size
        self.batch_size  = 256      # minibatch size
        self.gamma = 0.99       # discount factor
        self.tau = 1e-3         # for soft update of target parameters
        self.lr = 1e-3          # learning rate
        self.update_every = 1   # Learn every UPDATE_EVERY time steps.
        self.train_eps = 1000
        self.train_steps = 2000
        self.eval_eps = 1
        self.eval_steps = 2000
        self.eps_start = 1.0
        self.eps_decay = 0.995
        self.eps_end = 0.01
        self.hidden_layers = [64, 64]
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frames = 1

def env_agent_config(cfg:DQNConfig):
    """
    Create env and agent
    ------------------------
    Input: configuration
    Output: env, agent
    """
    if cfg.ShowImage:
        env = gym.make(cfg.env, render_mode="human")
    else:
        env = gym.make(cfg.env)
    
    # env.seed(cfg.seed)

    action_dim = env.action_space.n
    state_len = len(env.observation_space.shape)
    state_dim = 1
    for i in range(state_len):
        state_dim = state_dim * env.observation_space.shape[i]
    
    print(f"action dim: {action_dim}, state dim: {state_dim}")
    N = 2 if cfg.frames > 1 else 1
    agent = DQN(state_dim * N,action_dim,cfg)
    print(agent.qnetwork_local)
    return env,agent

def normalize(state, observation_space = None):
    return (state) / 255

def stack_frame(stacked_frames, frame, is_new, cfg:DQNConfig):
    if is_new:
        for i in range(cfg.frames):
            stacked_frames.append(frame)
    else:
        for i in range(cfg.frames - 1):
            stacked_frames[i] = stacked_frames[i+1]
        
        stacked_frames[-1] = frame
    # print(f"diff: {(stacked_frames[-1] - stacked_frames[0]) / 255}")
    return stacked_frames

def train(cfg:DQNConfig,env,agent):
    """
    Train model, and save agent and result during training.
    --------------------------------
    Input: configuration, env, agent
    Output: reward and ma_reward
    """
    print('Start to train !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward

    eps = cfg.eps_start
    best_reward = -10000
    for i_ep in range(cfg.train_eps):
        stacked_frames = []
        observation, info = env.reset()
        # stacked_frames = stack_frame(stacked_frames, observation, True, cfg)
        state = observation
        # state = normalize(np.concatenate(stacked_frames, axis=0))
        # state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
        ep_reward = 0
        i_step = 0
        
        while True:
        # for i_step in range(cfg.train_steps):
            action = agent.act(state, eps)
            next_observation, _, terminated, truncated, info = env.step(action)
            # print(next_observation,end="\r")

            reward = 0
            ## reward calculation: 在这里定义奖励函数，目前定义的奖励可能练不出来, 需要自己调整
            x_position = next_observation[0]  
            y_position = next_observation[1]  
            vx = next_observation[2]         
            vy = next_observation[3]          
            angle = next_observation[4]      
            angular_velocity = next_observation[5]  
            left_leg_contact = next_observation[6]  
            right_leg_contact = next_observation[7] 

            max_safe_vy = 0.3
            upper_altitude = 0.8
            down_altitude = 0.0
            safe_speed_threshold = 0.05
            thrust_applied = ( action == 2) 
            fuel_consumption = 1.0 if action != 0 else 0

            fuel_penalty = -fuel_consumption
            position_reward = -abs(x_position)
            horizontal_speed_reward = -abs(vx)
            attitude_reward = -abs(angle) - abs(angular_velocity)
            vertical_speed_reward = -abs(vy) if abs(vy) > max_safe_vy else 0
            vertical_speed_reward += -10 if vy > 0 else 0


            if y_position < upper_altitude and y_position > down_altitude and abs(vy) > safe_speed_threshold and thrust_applied:
                thrust_reward = 10
            else:
                thrust_reward = 0

            if left_leg_contact and right_leg_contact and abs(x_position) < 0.1:
                landing_reward = 100
            elif left_leg_contact and right_leg_contact:
                landing_reward = 20
            else:
                landing_reward = 0

            reward = (
                40 * attitude_reward +        # 姿态保持奖励
                40 * position_reward +         # 水平位置奖励
                1 * horizontal_speed_reward + # 水平速度奖励
                5 * vertical_speed_reward +   # 垂直速度奖励
                20 * thrust_reward +           # 推进器奖励
                20 * fuel_penalty +                # 燃料惩罚
                10 * landing_reward                # 成功着陆奖励
            )

            # time.sleep(0.1)
            
            


                

            # stacked_frames = stack_frame(stacked_frames, next_observation, False, cfg)
            next_state = next_observation
            # next_state = normalize(np.concatenate(stacked_frames, axis=0))
            # next_state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
            agent.step(state, action, reward, next_state, terminated or truncated)
            state = next_state
            ep_reward += reward
            i_step += 1

            # print(f"episode: {i_ep}, step: {i_step}", end="\r")
            # print(f"action: {action}, real_action: {action_in}")
            # print(f"state: {state}", end="\n")
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, eps: {eps}, step {i_step}, action: {action} Reward:{ep_reward:.3f}", end="\r")
            if terminated or truncated or (i_step >= cfg.train_steps):
                break

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 

        eps = max(cfg.eps_end, cfg.eps_decay * eps)
        if i_ep % 99 == 0:
            print('\nsave model')
            agent.save(cfg.model_path)
            np.savetxt(cfg.model_path+'reward_{}.txt'.format(curr_time),rewards)
            np.savetxt(cfg.model_path+'ma_reward_{}.txt'.format(curr_time),ma_rewards)

    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg:DQNConfig,env,agent):
    """
    Evaluate Model
    """
    print('Start to eval !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward

    for i_ep in range(cfg.eval_eps):
        stacked_frames = []
        observation, info = env.reset(seed=cfg.seed)
        # stacked_frames = stack_frame(stacked_frames, observation, True, cfg)
        state = observation
        # state = normalize(np.concatenate(stacked_frames, axis=0))
        # state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
        ep_reward = 0
        i_step = 0
        while True:
        # for i_step in range(cfg.eval_steps):
            action = agent.act(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            # stacked_frames = stack_frame(stacked_frames, next_observation, False, cfg)
            state = next_observation
            # state = normalize(np.concatenate(stacked_frames, axis=0))
            # state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
            ep_reward += reward
            i_step += 1

            print(f"Episode:{i_ep+1}/{cfg.eval_eps}, action: {action}, step {i_step} Reward:{ep_reward:.3f}", end="\r")
            # print(f"\rreward: {reward}", end="")
            # print(f"action: {action}, real_action: {action_in}")
            if terminated or truncated or (i_step >= cfg.eval_steps):
                break

        print(f"\nEpisode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
    np.savetxt(cfg.model_path+'reward_eval.txt',rewards)
    print('Complete evaling！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg=DQNConfig()
    
    env,agent = env_agent_config(cfg)
    if cfg.train:
        # train
        if cfg.load_model:
            print(">>>>>>>>>>load model<<<<<<<<<<<<<<<")
            agent.load(path=cfg.model_path)
        rewards, ma_rewards = train(cfg, env, agent)
    else:
        # eval
        print(">>>>>>>>>>load model<<<<<<<<<<<<<<<")
        agent.load(path=cfg.model_path)
        rewards, ma_rewards = eval(cfg, env, agent)
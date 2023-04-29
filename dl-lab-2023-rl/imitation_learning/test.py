from __future__ import print_function

import sys
sys.path.append("./") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
import torch.nn.functional as F

from agent.bc_agent import BCAgent
from utils import *

act_hist = np.empty(100, dtype=np.int32)
print(len(act_hist))
act_hist.fill(-1)
act_hist_len = len(act_hist)
frz_count = 0
frz_step = 0
counter = 0

def check_and_unfreeze(action):
    global frz_count
    for act in act_hist:
        if np.all(act_hist == -1):
            return True
        if not np.all(act_hist == act):
            return False
        if np.all(act_hist == 3):
            return False
    frz_count += 1
    print("Freeeeeeeeeeeeeeezeeeeeeeee!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ", frz_count)
    return True

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...

        state = rgb2gray(state)
        state = np.expand_dims(state, (0, 1))
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        act = F.softmax(agent.predict(state))
        print(torch.argmax(act))
        act = torch.argmax(act).item()
        # a = id_to_action(act)

        global counter
        global frz_step

        if counter == act_hist_len:
            counter = 0
        if check_and_unfreeze(act):
            frz_step = step
        print(act)

        act_hist[counter] = act
        counter += 1

        a = id_to_action(act)

        unique, counts = np.unique(act_hist, return_counts = True)
        print(dict(zip(unique, counts)))

        if step<=15:
            a = np.array([0.0, 1.0, 0.0])

        if step>=20 and step<=80 and  np.all(act_hist[:8] == 3):
            a = np.array([0.0, 0.0, 0.0])

        if step<=(frz_step+5):a = np.array([0.0, 1.0, 0.0])

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    # agent = BCAgent(...)
    # agent.load("models/bc_agent.pt")
    agent = BCAgent()
    agent.load("models/agent.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

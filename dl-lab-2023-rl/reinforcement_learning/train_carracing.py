# export DISPLAY=:0 

import sys
sys.path.append("./") 

import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent
import torch
from imitation_learning.agent.networks import CNN
from torchsummary import summary


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length))
    state = np.array(image_hist).reshape(history_length, 96, 96)
    # print(state.shape)
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        action_id = agent.act(state=state, deterministic=deterministic, env='CarRacing')
        action = id_to_action(action_id)

        if do_training:
            print(str(action_id) + ' ', end='')


        # Hint: frame skipping might help you to get better results.
        if not do_training:
            if action_id == 2:
                action_id = int(np.random.choice([0, 2], p= [0.10, 0.90]))
            action = id_to_action(action_id)

        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1
    print()

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name='CarRacingDQN', stats=["episode_reward", "straight", "left", "right", "accel", "brake"])

    best_eval_reward = 0.0
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        max_timesteps = 300
        epsilon = 1.0

        if i > 300:
            epsilon *= 0.995
            max_timesteps = int(1.2*i)

        if epsilon < 0.1: epsilon = 0.1
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering = True, skip_frames=3, max_timesteps=max_timesteps, history_length=history_length)
        
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        # store model.
        if i % eval_cycle == 0:
            print("Evaluating Model")
            avg_eval_reward = 0.0
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length)
                avg_eval_reward = (avg_eval_reward*j + stats.episode_reward)/(j + 1)
            print("Evaluation Reward for {:.4f}".format(avg_eval_reward))
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                agent.save(os.path.join(model_dir, "dqn_agent_best.pt"))
                print("Model saved")
        agent.save(os.path.join(model_dir, "dqn_agent_final.pt"))

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 10
    num_actions = 5

    env = gym.make('CarRacing-v0').unwrapped
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    Q_network = CNN(history_length=history_length, n_classes=num_actions).cuda()
    Q_target = CNN(history_length=history_length, n_classes=num_actions).cuda()

    summary(Q_network, (history_length, 96, 96))

    agent = DQNAgent(Q = Q_network, Q_target=Q_target, num_actions=num_actions, 
                     device=device, lr=3e-4, history_length=history_length)
    
    train_online(env, agent, num_episodes=1000, history_length=history_length, model_dir="./models_carracing")


import numpy as np
import matplotlib.pyplot as plt
import random

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, np.float32(0.2)]): return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])
    

def show_hist(Y, save):
    plt.figure()
    values, count = np.unique(np.array(Y), return_counts = True)
    Y_dict = dict(zip(values, count))
    print(Y_dict)
    plt.bar(Y_dict.keys(), Y_dict.values(), color = 'g')
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("Action id")
    plt.ylabel("Number of samples")
    plt.title(save)
    plt.savefig(f'./figs/Histogram_{save}.png', transparent = False, facecolor = 'white')
    plt.show()

def balance_actions(X, y, drop):
    num_st = y.count(0)
    # print(num_st)
    # print(type(X_train), type(y_train))
    st = random.sample([i for i, j in enumerate(y) if j == 0], int(drop*num_st))
    st.sort()
    # print(len(st))
    y_b = []
    X_b = np.empty(shape=(X.shape[0] - len(st), X.shape[1], X.shape[2]))
    # print(len(X_b))
    iter = 0
    i = 0
    for i in range(len(X)):
        if i in st:
            continue
        y_b.append(y[i])
        X_b[iter] = X[i]
        iter += 1
    return X_b, y_b
    

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))

from __future__ import print_function

import sys
sys.path.append("./") # changed from sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
import logging
from tqdm import tqdm
from tqdm import trange

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

logging.basicConfig(level = logging.INFO)

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    print(os.path)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=0):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    y_train = [action_to_id(i) for i in y_train]
    y_valid = [action_to_id(i) for i in y_valid]
    # print(X_train.shape, X_valid.shape)
    # X_train = np.reshape(np.expand_dims(X_train, 1), (len(X_train) - history_length + 1, history_length, X_train.shape[-2], X_train.shape[-1]))
    X_train = np.array([X_train[frame - history_length : frame] for frame in range(history_length, X_train.shape[0] + 1) ])
    # print(X_train.shape, type(X_train))
    X_valid = np.array([X_valid[frame - history_length : frame] for frame in range(history_length, X_valid.shape[0] + 1) ])
    # X_valid = np.reshape(np.expand_dims(X_valid, 1), (len(X_valid) - history_length + 1, history_length, X_valid.shape[-2], X_valid.shape[-1]))
    y_train = np.array(y_train[history_length - 1:])
    y_valid = np.array(y_valid[history_length - 1:])
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, history_length, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 
 
    # print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    # agent = BCAgent(...)
    agent = BCAgent(lr, history_length)
    
    tensorboard_eval = Evaluation(tensorboard_dir, "Run1", stats = ["train_loss", "train_acc", "val_loss", "val_acc"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)

    best_val_acc = 0.0
    logging.info('Training the model:')
    def sample_minibatch(X, y, grad = True):
        loss, acc = agent.update(X, y, grad)
        return loss, acc


    avg_train_loss = 0.0
    avg_train_acc = 0.0
    logging.info('#' * 50)
    i = 0
    t = trange(n_minibatches, desc='')
    # loss_list = []
    # acc_list = []
    for iter in t: #range(n_minibatches):
        frame_num = np.random.randint(0, len(X_train), batch_size)
        # print(X_train.shape, y_train.shape)
        X_batch = X_train[frame_num]
        y_batch = y_train[frame_num]
        train_loss, train_acc = sample_minibatch(X_batch, y_batch)
        # loss_list.append(train_loss)
        # acc_list.append(train_acc)
        avg_train_loss = (avg_train_loss*i + train_loss)/(i + 1)
        avg_train_acc = (avg_train_acc*i + train_acc)/(i + 1)
        # logging.info('Training Loss per batch: %f', train_loss)
        # logging.info('Training Accuracy per batch: %f', train_acc)
        i+=1

        b = 0
        vi = 0
        avg_val_loss = 0.0
        avg_val_acc = 0.0
        for v_batch in range(int(len(X_valid)/batch_size)):
            v_frames = np.random.randint(0, len(X_valid), batch_size)
            # v_frames = np.arange(b, b + batch_size, 1)
            X_batch_v = X_valid[v_frames]
            y_batch_v = y_valid[v_frames]
            val_loss, val_acc = sample_minibatch(X_batch_v, y_batch_v, False)
            avg_val_loss = (avg_val_loss*vi + val_loss)/(vi + 1)
            avg_val_acc = (avg_val_acc*vi + val_acc)/(vi + 1)
            # t.set_description('Validation Loss = {:.4f}, Validation Accuracy = {:.4f}'.format(avg_val_loss, 
            #                                                                             avg_val_acc))
            b += batch_size
            vi += i

        t.set_description('Training Loss = {:.4f}, Training Accuracy = {:.4f}, Validation Loss = {:.4f}, Validation Accuracy = {:.4f}'.format(
            avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))

        eval_dict = {
            "train_loss" : avg_train_loss,
            "train_acc" : avg_train_acc,
            "val_loss" : avg_val_loss,
            "val_acc" : avg_val_acc
        }
        if i % 10 == 0:
            tensorboard_eval.write_episode_data(i, eval_dict)
            agent.save(os.path.join(model_dir, "agent.pt"))
            # print("Model saved in file: %s" % model_dir)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            # agent.save(os.path.join(model_dir, "agent.pt"))
            # print("Model saved in file: %s" % model_dir)
        # agent.save(os.path.join(model_dir, "agent.pt"))
        # print("Model saved in file: %s" % model_dir)



    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)
    # agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)

if __name__ == "__main__":

    hist_len = 1
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length = hist_len)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, history_length = hist_len, n_minibatches=10000, batch_size=64, lr=3e-4)
 

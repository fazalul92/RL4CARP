"""@package RL4CARP
Main File for testing pre-trained models
Creates a Planner instance and loads pre-trained weights. Runs the simulation on test dataset.
Multiple arguments for training can be sent at runtime.

seed            : Random seed
checkpoint      : Link to the actor/critic checkpoints (REQUIRED)
test            : Flag to set the mode to Testing instead of Training.
test_dir        : Directory for storing sample result images.
nodes           : Number of nodes to train for (Can be 10, 20, 50 or 100)
actor_lr        : Learning rate for the actor
critic_lr       : Learning rate for the critic
max_grad_norm   : Value for clipping the gradient
batch_size      : Training/Testing batch size
hidden          : Number of hidden layers to be generated
dropout         : Value for dropout between NN layers
layers          : Number of layers in the NN
train_size      : Number of examples to be trained on
valid_size      : Number of examples to be validated on
epochs          : Number of epochs to train for
"""

import argparse
from planner import Planner

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--test_dir', default='test', type=str)
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--epochs', default=33, type=int)

    args = parser.parse_args()
    planner = Planner(args)

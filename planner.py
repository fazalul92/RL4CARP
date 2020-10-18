"""@package RL4CARP
Class file for Planner
Initiates Actor and Critic
Loads weights from custom checkpoints if specified
Runs training or testing based on the passed parameters
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DRL4TSP
from critic import StateCritic
from tasks import vrp
from tasks.vrp import VehicleRoutingDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestError(Exception):
    """
    An error that can be raised for Exceptions during Testing

    Parameters:
    -----------
    message: str
        The message to be shown to the user
    """

    def __init__(self, message="A valid checkpoint for the trained model is required for testing. Use the trainer or load pre-trained weights."):
        self.message = message
        super().__init__(self.message)

class Planner:
    """ The Planner through which the training and testing is carried out.

    Parameters:
    -----------
    args: argumentparser
        Includes all the arguments that are being used when the training/testing is started

    Methods:
    --------
    train(actor, critic, num_nodes, train_data, valid_data, reward_fn, render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, epochs, **kwargs)
        Trains the model based on the parameters passed
    
    validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5)
        Validates the model after initial training or tests the model.

    """
    def __init__(self, args):
        # Setting maximum capacity for vehicles based on the number of nodes chosen.
        LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
        MAX_DEMAND = 9
        STATIC_SIZE = 2 # (x, y)
        DYNAMIC_SIZE = 2 # (load, demand)

        max_load = LOAD_DICT[args.num_nodes]

        train_data = VehicleRoutingDataset(args.train_size, args.num_nodes, max_load, MAX_DEMAND, args.seed)
        valid_data = VehicleRoutingDataset(args.valid_size, args.num_nodes, max_load, MAX_DEMAND, args.seed + 1)

        actor = DRL4TSP(STATIC_SIZE,  DYNAMIC_SIZE, args.hidden_size, train_data.update_dynamic, train_data.update_mask, args.num_layers, args.dropout).to(device)
        critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

        kwargs = vars(args)
        kwargs['train_data'] = train_data
        kwargs['valid_data'] = valid_data
        kwargs['reward_fn'] = vrp.reward
        kwargs['render_fn'] = vrp.render
        
        if args.checkpoint:
            path = os.path.join(args.checkpoint, 'actor.pt')
            actor.load_state_dict(torch.load(path, device))

            path = os.path.join(args.checkpoint, 'critic.pt')
            critic.load_state_dict(torch.load(path, device))

        if not args.test:
            self.train(actor, critic, **kwargs)
        elif not args.checkpoint:
            raise TestError()
        test_data = VehicleRoutingDataset(args.valid_size, args.num_nodes, max_load, MAX_DEMAND, args.seed + 2)

        test_dir = args.test_dir
        print("Saving test results to {}".format(test_dir))
        test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
        out = self.validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

        print('Average tour length: ', out)

    def validate(self, data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
        """Used to monitor progress on a validation set & optionally plot solution.
        
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            used for loading data in batches
        actor: model.DRL4TSP
            the Actor model which decides the next decision to take
        reward_fn: func
            function for calculating the reward
        render_fn: func, optional
            function for rendering the results on a 2D graph
        save_dir: str
            directory to which the renders have to be stored
        num_plot: int, optional
            number of plots to be included in a graph
        """

        actor.eval()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        rewards = []
        for batch_idx, batch in enumerate(data_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            with torch.no_grad():
                tour_indices, _ = actor.forward(static, dynamic, x0)

            reward = reward_fn(static, tour_indices).mean().item()
            rewards.append(reward)

            if render_fn is not None and batch_idx < num_plot:
                name = 'batch%d_%2.4f.png'%(batch_idx, reward)
                path = os.path.join(save_dir, name)
                render_fn(static, tour_indices, path)

        actor.train()
        return np.mean(rewards)



    def train(self, actor, critic, num_nodes, train_data, valid_data, reward_fn,
            render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, epochs,
            **kwargs):
        """Constructs the main actor & critic networks, and performs all training.
        
        Parameters
        ----------
        actor: model.DRL4TSP
            The actor network that decides which nodes to visit
        critic: critic.StateCritic
            The critic that estimates the log probabilities of next states
        kwargs: argumentparser
            All the other inputs are from the initial training or testing arguments provided
        """

        task = 'vrp'
        now = '%s' % datetime.datetime.now().time()
        now = now.replace(':', '_')
        save_dir = os.path.join(task, '%d' % num_nodes, now)

        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

        train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

        best_params = None
        best_reward = np.inf

        for epoch in range(epochs):

            actor.train()
            critic.train()

            times, losses, rewards, critic_rewards = [], [], [], []

            epoch_start = time.time()
            start = epoch_start

            for batch_idx, batch in enumerate(train_loader):

                static, dynamic, x0 = batch

                static = static.to(device)
                dynamic = dynamic.to(device)
                x0 = x0.to(device) if len(x0) > 0 else None


                # Full forward pass through the dataset
                tour_indices, tour_logp = actor(static, dynamic, x0)


                # Sum the log probabilities for each city in the tour
                reward = reward_fn(static, tour_indices)

                # Query the critic for an estimate of the reward
                critic_est = critic(static, dynamic).view(-1)

                advantage = (reward - critic_est)
                actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
                critic_loss = torch.mean(advantage ** 2)

                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()

                critic_rewards.append(torch.mean(critic_est.detach()).item())
                rewards.append(torch.mean(reward.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())

                if (batch_idx + 1) % 100 == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end

                    mean_loss = np.mean(losses[-100:])
                    mean_reward = np.mean(rewards[-100:])

                    print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                        (batch_idx, len(train_loader), mean_reward, mean_loss,
                        times[-1]))

            mean_loss = np.mean(losses)
            mean_reward = np.mean(rewards)

            # Save the weights
            epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(epoch_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

            # Save rendering of validation set tours
            valid_dir = os.path.join(save_dir, '%s' % epoch)

            mean_valid = self.validate(valid_loader, actor, reward_fn, render_fn,
                                valid_dir, num_plot=5)

            # Save best model parameters
            if mean_valid < best_reward:

                best_reward = mean_valid

                save_path = os.path.join(save_dir, 'actor.pt')
                torch.save(actor.state_dict(), save_path)

                save_path = os.path.join(save_dir, 'critic.pt')
                torch.save(critic.state_dict(), save_path)

            print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
                '(%2.4fs / 100 batches)\n' % \
                (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
                np.mean(times)))
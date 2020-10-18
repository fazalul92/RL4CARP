"""@package RL4CARP
Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each node has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a predefined capacity, the must visit all nodes
    3. When the vehicle load is 0, it must return to the depot to refill
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg') # Avoids plotting when running in the cluster
import matplotlib.pyplot as plt


class VehicleRoutingDataset(Dataset):
    """Simulated Dataset Generator
    This class can generate random points in euclidean space for 
    training and testing the reinforcement learning agent.

    ...

    Parameters
    ----------
    num_samples : int
        number of training/testing examples to be generated
    input_size  : int
        number of nodes to be generated in each training example
    max_load    : int
        maximum load that a vehicle can carry
    max_demand  : int
        maximum demand that a node can have
    seed        : int
        random seed for reproducing results

    Methods
    -------
    __len__()
        To be used with class instances. class_instance.len returns the num_samples value
    
    __getitem__(idx)
        returns the specific example at the given index (idx)
    
    update_mask(mask, dynamic, chosen_idx=None):
        updates the generated mask to hide any invalid states
    
    update_dynamic(dynamic, chosen_idx):
        updates the loads and demands for the input index (chosen_idx)
    """

    def __init__(self, num_samples,input_size, max_load=20, max_demand=9,
                 seed=None):
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        # Depot location will be the first node in each
        locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = locations

        # All states will broadcast the drivers current load
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load.
        demands = torch.randint(1, max_demand + 1, dynamic_shape,dtype=torch.float)
        demands = demands / float(max_load)

        demands[:, 0, 0] = 0  # depot starts with a demand of 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1),dtype=torch.float)

    def __len__(self):
        """Returns the number of examples being trained/tested on"""
        return self.num_samples

    def __getitem__(self, idx):
        """Returns the specific example at the given index (idx)
        
        Parameters
        ----------
        idx : int
            index for which the example has to be returned.
        """
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide any non-valid states

        Parameters
        ----------
        mask: torch.Tensor
            current mask to which the update has to be made
        dynamic:  torch.Tensor
            dynamic values to be updated
        chosen_idx: torch.Tensor
            the chosen indices where the update has to be made
        """

        loads = dynamic.data[:, 0] 
        demands = dynamic.data[:, 1]

        if demands.eq(0).all():
            return demands * 0.

        new_mask = demands.ne(0) * demands.lt(loads)
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (~repeat_home).any():
            new_mask[(~repeat_home).nonzero(), 0] = 0.

        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updating the loads and demands at the given index

        Parameters
        ----------
        dynamic: torch.Tensor
            new demand values to be used
        chosen_idx: torch.Tensor
            the indices at which the update has to be made
        """

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Satisfy as much demand as possible once a city is visited.
        if visit.any():

            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)

        return tensor.clone().detach().to(dynamic.device)


def reward(static, tour_indices):
    """Calculates the euclidean distance between all nodes given by tour_indices

    Parameters
    ----------
    static: torch.Tensor
        The (x,y) coordinates of the tour indices
    
    tour indices: torch.Tensor
        The indices in the main example tensor where the (x,y) coordinates are located.
    """

    # To convert the indices back to a complete tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensuring that the first and last nodes are the depot locations
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Calculating euclidean distances between every two consecutive points.
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1)


def render(static, tour_indices, save_path):
    """Function to plot the calculated near optimal solution

    Parameters
    ----------
    static: torch.Tensor
        The (x,y) coordinates of the tour indices
    
    tour indices: torch.Tensor
        The indices in the main example tensor where the (x,y) coordinates are located.

    save_path: str
        The location for saving the rendered solutions
    """

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots, sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
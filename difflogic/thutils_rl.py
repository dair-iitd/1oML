#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for PyTorch."""

import torch
import torch.nn.functional as F
import numpy as np

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from IPython.core.debugger import Pdb
__all__ = [
    'binary_accuracy', 'rms', 'monitor_saturation', 'monitor_paramrms',
    'monitor_gradrms'
]

def check_validity(grid, constraints=None):
    grid = grid.cpu().numpy()
    constraints = constraints.cpu().numpy()
    grid = grid.argmax(axis=2)
    for x in range(len(grid)):
        row = set(grid[x])
        if len(row) != len(grid):
            return False
        col = set(grid[:, x])
        if len(col) != len(grid):
            return False
    if constraints is None:
        return True
    gt = zip(*np.nonzero(constraints[0]))
    for ind in gt:
        next_ind = (ind[0], ind[1]+1)
        if grid[next_ind] > grid[ind]:
            return False
    lt = zip(*np.nonzero(constraints[1]))
    for ind in lt:
        next_ind = (ind[0], ind[1]+1)
        if grid[next_ind] < grid[ind]:
            return False
    return True

def match_query(query, pred):
    mask = (query > 0)
    return torch.equal(query[mask], pred[mask])

def is_safe_nqueens(grid):
    size = int(len(grid)**0.5)

    grid = grid.reshape(size, size)
    indices = torch.nonzero(grid)
    if len(indices) != size:
        return False
    for x in range(size):
        r1, c1 = indices[x]
        for y in range(x+1, size):
            r2, c2 = indices[y]
            if (r1 == r2) or (c1 == c2) or (torch.abs(r1-r2) == torch.abs(c1-c2)):
                return False
    return True

def is_safe_futoshiki(grid, constraints):
    size = int(len(grid)**0.3334)
    grid = grid.reshape(size, size, size).float()
    gold = torch.ones(size, size).cuda()
    if torch.sum(torch.abs(grid.sum(dim=0)-gold)) > 0:
        return False
    if torch.sum(torch.abs(grid.sum(dim=1)-gold)) > 0:
        return False
    if torch.sum(torch.abs(grid.sum(dim=2)-gold)) > 0:
        return False

    constraints = constraints.transpose(0, 1)
    constraints = constraints.reshape(2, size, size, size)
    constraints = constraints[:, :, :, 0]
    return check_validity(grid, constraints)


def is_safe_sudoku(x, n):
    grid = x.detach().cpu().numpy().astype(int)
    grid = grid.reshape(n, n)

    b_size = int(np.sqrt(n))

    for i in range(n):
        if len(set(grid[i])) < n:
            return False
        if len(set(grid[:, i])) < n:
            return False

        b_row = i//b_size
        b_col = i % b_size

        if len(set(grid[b_size*b_row:b_size*(b_row+1), b_size*b_col:b_size*(b_col+1)].flatten())) < n:
            return False
    return True


def instance_accuracy(label, raw_pred, return_float=True, feed_dict=None, task='nqueens', args=None, inputs=None):
    with torch.no_grad():
        def compare_func_futo(x, query): return match_query(
            query[:, 0].float(), x) and is_safe_futoshiki(x, query[:, 1:])

        def compare_func_nqueens(x, query): return match_query(
            query[:, 0].float(), x) and is_safe_nqueens(x)
        def compare_func_sudoku(x, query): return match_query(query, x) and is_safe_sudoku(
            x, args.test_number_begin if return_float else args.train_number)

        compare_func = compare_func_nqueens
        if task == 'futoshiki':
            compare_func = compare_func_futo
        elif task == 'sudoku':
            compare_func = compare_func_sudoku
        #
        if (args.is_encoder_decoder) and (inputs is not None) and (feed_dict['target_set'].shape[0] != raw_pred.shape[0]): 
            return _instance_accuracy_encoder_decoder(label, raw_pred, compare_func, return_float, feed_dict, args, inputs=inputs)
        else:
            return _instance_accuracy(label, raw_pred, compare_func, return_float, feed_dict, args, inputs=inputs)

def _instance_accuracy_encoder_decoder(label, raw_pred, compare_func, return_float=True, feed_dict=None, args=None, inputs=None):
    """get instance-wise accuracy for structured prediction task instead of pointwise task"""
    # disctretize output predictions
    if not args.task_is_sudoku:
        pred = as_tensor(raw_pred)
        pred = (pred > 0.5).float()
    else:
        step_pred = as_tensor(raw_pred.argmax(dim=1)).float()
        pred = step_pred[:, :, -1]

        # step pred is batch_size x 81 x num_steps
        # transpose for more efficient reward calculation
        # new shape is batch_size x num_Steps x 81
        step_pred = step_pred.transpose(1, 2)

    label = as_tensor(label).type(pred.dtype)

    diff = (label == pred)
    point_acc = torch.sum(diff).float()/label.numel()
    incorrect = torch.min(diff, dim=1)[0]
    in_acc = torch.sum(incorrect).float()/len(label)

    errors = []
    corrected_acc = 0
    reward = []
    acc_vector = []
    for i, x in enumerate(pred):
        this_input = inputs[i].type(x.dtype)
        if compare_func(x, this_input):
            corrected_acc += 1
            acc_vector.append(1)
        else:
            acc_vector.append(0)
        if args.task_is_sudoku:
            #TODO compute reward fn for enocder decoder. Adding dummy reward 0 
            # reward computation can happen only in inference when yhat is computed without teacher forcing
            #reward.append(torch.tensor(0.0))
            
            # if args.use_gpu:
            #    diff = torch.zeros(len(feed_dict['target_set'][i]),step_pred.shape[1], device=torch.device("cuda"))
            # else:
            #    diff = torch.zeros(len(feed_dict['target_set'][i]),step_pred.shape[1]).cuda()
            # for target_idx,target in enumerate(feed_dict['target_set'][i,:feed_dict['count'][i]].float()):
            #    diff[target_idx] = torch.sum(~(step_pred[i]==target), dim=1).float()
            # for target_idx in range(feed_dict['count'][i],diff.shape[0]):
            #    diff[target_idx] = diff[target_idx-1]
            #
            # alternative tensor way
            if feed_dict['target_set'].shape[0] == inputs.shape[0]: 
                NS, NN, TS = step_pred.size(1), step_pred.size(
                    2), feed_dict['target_set'].size(1)
                diff = (step_pred[i].unsqueeze(-1).expand(NS, NN, TS).transpose(0, 2).float() !=
                        feed_dict['target_set'][i].unsqueeze(-1).expand(TS, NN, NS).float()).sum(dim=1).float()

                if args.rl_reward == 'count':
                    reward.append(diff.mean(dim=1))
                else:
                    reward.append(torch.clamp_max(diff, 1).mean(dim=1))
        else:
            raise
            diff = torch.sum(
                ~(feed_dict["target_set"][i].type(x.dtype) == x), dim=1).float()
            if args.rl_reward == 'count':
                reward.append(diff)
            else:
                reward.append(torch.clamp_max(diff, 1))
    corrected_acc /= len(pred)

    if feed_dict['target_set'].shape[0] == inputs.shape[0]:
        reward = -torch.stack(reward)
    else:
        reward = torch.zeros_like(feed_dict['mask'])

    ##############target_set_accuracy = (reward.max(dim=1)[0] >= 0).float().mean()
    target_set_accuracy = torch.tensor(0.0)
    if return_float:
        return {"accuracy": in_acc.item(),
                "corrected accuracy": corrected_acc,
                "pointwise accuracy": point_acc.item(),
                "target set accuracy": target_set_accuracy.item()}, errors, reward  # , acc_vector
    return {"accuracy": torch.tensor(in_acc),
            "corrected accuracy": torch.tensor(corrected_acc),
            "pointwise accuracy": point_acc,
            "target set accuracy": target_set_accuracy}, errors, reward



def _instance_accuracy(label, raw_pred, compare_func, return_float=True, feed_dict=None, args=None, inputs=None):
    """get instance-wise accuracy for structured prediction task instead of pointwise task"""
    # disctretize output predictions
    if not args.task_is_sudoku:
        pred = as_tensor(raw_pred)
        pred = (pred > 0.5).float()
    else:
        step_pred = as_tensor(raw_pred.argmax(dim=1)).float()
        pred = step_pred[:, :, -1]

        # step pred is batch_size x 81 x num_steps
        # transpose for more efficient reward calculation
        # new shape is batch_size x num_Steps x 81
        step_pred = step_pred.transpose(1, 2)

    label = as_tensor(label).type(pred.dtype)

    diff = (label == pred)
    point_acc = torch.sum(diff).float()/label.numel()
    incorrect = torch.min(diff, dim=1)[0]
    in_acc = torch.sum(incorrect).float()/len(label)

    errors = []
    corrected_acc = 0
    reward = []
    new_targets = []
    acc_vector = []
    for i, x in enumerate(pred):
        input = feed_dict['query'][i].type(x.dtype) if inputs is None else inputs[i].type(x.dtype)
        if compare_func(x, input):
            corrected_acc += 1
            acc_vector.append(1)
            # check if pred matches any target
            if ((feed_dict['target_set'][i].type(x.dtype) == x).sum(dim=1) == x.shape[0]).sum() > 0:
                new_targets.append((None, None))
            else:
                new_targets.append((x, 0))
        else:
            acc_vector.append(0)
            errors.append(feed_dict["count"][i].item())
            new_targets.append((None, None))
        if args.task_is_sudoku:
            # if args.use_gpu:
            #    diff = torch.zeros(len(feed_dict['target_set'][i]),step_pred.shape[1], device=torch.device("cuda"))
            # else:
            #    diff = torch.zeros(len(feed_dict['target_set'][i]),step_pred.shape[1]).cuda()
            # for target_idx,target in enumerate(feed_dict['target_set'][i,:feed_dict['count'][i]].float()):
            #    diff[target_idx] = torch.sum(~(step_pred[i]==target), dim=1).float()
            # for target_idx in range(feed_dict['count'][i],diff.shape[0]):
            #    diff[target_idx] = diff[target_idx-1]
            #
            # alternative tensor way
            NS, NN, TS = step_pred.size(1), step_pred.size(
                2), feed_dict['target_set'].size(1)
            diff = (step_pred[i].unsqueeze(-1).expand(NS, NN, TS).transpose(0, 2).float() !=
                    feed_dict['target_set'][i].unsqueeze(-1).expand(TS, NN, NS).float()).sum(dim=1).float()

            if args.rl_reward == 'count':
                reward.append(diff.mean(dim=1))
            else:
                reward.append(torch.clamp_max(diff, 1).mean(dim=1))
        else:
            diff = torch.sum(
                ~(feed_dict["target_set"][i].type(x.dtype) == x), dim=1).float()
            if args.rl_reward == 'count':
                reward.append(diff)
            else:
                reward.append(torch.clamp_max(diff, 1))
    corrected_acc /= len(pred)

    reward = -torch.stack(reward)
    target_set_accuracy = (reward.max(dim=1)[0] >= 0).float().mean()

    if return_float:
        return {"accuracy": in_acc.item(),
                "corrected accuracy": corrected_acc,
                "pointwise accuracy": point_acc.item(),
                "target set accuracy": target_set_accuracy.item()}, errors, reward  # , acc_vector
    return {"accuracy": torch.tensor(in_acc),
            "corrected accuracy": torch.tensor(corrected_acc),
            "pointwise accuracy": point_acc,
            "target set accuracy": target_set_accuracy}, errors, reward


def instance_accuracy_nqueens(label, raw_pred, return_float=True, feed_dict=None, args=None):
    """get instance-wise accuracy for structured prediction task instead of pointwise task"""

    pred = as_tensor(raw_pred)
    pred = (pred > 0.5).float()

    label = as_tensor(label).float()
    diff = torch.abs(label-pred)
    point_acc = 1 - torch.sum(diff)/label.numel()
    incorrect_count = torch.sum(diff, dim=1)
    incorrect = len(torch.nonzero(incorrect_count))

    in_acc = 1-incorrect/len(label)

    errors = []
    reward = []
    corrected_acc = 0
    for i, x in enumerate(pred):
        if match_query(feed_dict["query"][i][:, 0].float(), x) and is_safe_nqueens(x):
            corrected_acc += 1
            acc_vector[i] = 1.0
        else:
            errors.append(feed_dict["count"][i].item())

        diff = torch.sum(
            torch.abs(feed_dict["target_set"][i].float()-x), dim=1)
        reward.append(diff)

    corrected_acc /= len(pred)

    reward = torch.stack(reward)
    if args is not None and args.rl_reward == 'count':
        reward = -1*reward.float()
    else:
        reward = -1*torch.clamp(reward, 1).float()

    if return_float:
        return {"accuracy": in_acc,
                "corrected accuracy": corrected_acc,
                "pointwise accuracy": point_acc.item(),
                "classification accuracy": classification_acc.item()}, errors, reward
    return {"accuracy": torch.tensor(in_acc),
            "corrected accuracy": torch.tensor(corrected_acc),
            "pointwise accuracy": point_acc,
            "classification accuracy": classification_acc}, errors, reward


def binary_accuracy(label, raw_pred, eps=1e-20, return_float=True):
    """get accuracy for binary classification problem."""
    pred = as_tensor(raw_pred).squeeze(-1)
    pred = (pred > 0.5).float()
    label = as_tensor(label).float()
    # The $acc is micro accuracy = the correct ones / total
    acc = label.eq(pred).float()

    # The $balanced_accuracy is macro accuracy, with class-wide balance.
    nr_total = torch.ones(
        label.size(), dtype=label.dtype, device=label.device).sum(dim=-1)
    nr_pos = label.sum(dim=-1)
    nr_neg = nr_total - nr_pos
    pos_cnt = (acc * label).sum(dim=-1)
    neg_cnt = acc.sum(dim=-1) - pos_cnt
    balanced_acc = ((pos_cnt + eps) / (nr_pos + eps) + (neg_cnt + eps) /
                    (nr_neg + eps)) / 2.0

    # $sat means the saturation rate of the predication,
    # measure how close the predections are to 0 or 1.
    sat = 1 - (raw_pred - pred).abs()
    if return_float:
        acc = as_float(acc.mean())
        balanced_acc = as_float(balanced_acc.mean())
        sat_mean = as_float(sat.mean())
        sat_min = as_float(sat.min())
    else:
        sat_mean = sat.mean(dim=-1)
        sat_min = sat.min(dim=-1)[0]

    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'satuation/mean': sat_mean,
        'satuation/min': sat_min,
    }


def rms(p):
    """Root mean square function."""
    return as_float((as_tensor(p)**2).mean()**0.5)


def monitor_saturation(model):
    """Monitor the saturation rate."""
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


def monitor_paramrms(model):
    """Monitor the rms of the parameters."""
    monitors = {}
    for name, p in model.named_parameters():
        monitors['paramrms/' + name] = rms(p)
    return monitors


def monitor_gradrms(model):
    """Monitor the rms of the gradients of the parameters."""
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['gradrms/' + name] = (rms(p.grad) / max(rms(p), 1e-8))
    return monitors

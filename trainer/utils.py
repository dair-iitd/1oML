import math
import time
import os, sys
import numpy as np
import torch
import torch.nn as nn
from IPython.core.debugger import Pdb

def compute_param_norm(parameters, norm_type= 2): 
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0 
    for p in parameters:
        if p.is_sparse:
            # need to coalesce the repeated indices before finding norm
            grad = p.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def compute_grad_norm(parameters, norm_type= 2): 
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0 
    for p in parameters:
        if p.grad.is_sparse:
            # need to coalesce the repeated indices before finding norm
            grad = p.grad.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm 

#copied from allennlp.trainer.util
def sparse_clip_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.is_sparse:
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


# === performing gradient descent
#copied from allennlp.trainer.util
def rescale_gradients(model, grad_norm = None):
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None

#called after loss.backward and optimizer.zero_grad. Before optimizer.step()
def gradient_normalization(model, grad_norm = None):
    # clip gradients
    #grad norm before clipping
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm_before_clip = compute_grad_norm(parameters)
    grad_norm_after_clip = grad_norm_before_clip
    param_norm_before_clip = compute_param_norm(parameters)
    grad_before_rescale = rescale_gradients(model, grad_norm)
    
    #torch.nn.utils.clip_grad_norm(model.parameters(), clip_val)
    grad_norm_after_clip = compute_grad_norm(parameters)
    #
    return grad_norm_before_clip.item(), grad_norm_after_clip.item(), param_norm_before_clip.item() 

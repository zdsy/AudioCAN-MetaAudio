"""
This file contains the following:
    -> Main proto net episodes for fixed and variable length sets
    -> Pairwise distance calculator
    -> Prototype calculator 
"""
##############################################################################
# IMPORTS
##############################################################################
import sys
import torch
import numpy as np
import torch.nn as nn
from typing import Callable
from torch.nn import Module
from torch.optim import Optimizer

from metrics import catagorical_accuracy, vote_catagorical_acc, majority_vote
from model.utils import mask_for_query, one_hot
from model.losses import CrossEntropyLoss


##############################################################################
# FIXED PROTONET EPISODE
##############################################################################
def proto_step_fixed(device, model, optimiser, loss_fn, x, y, pid, k_shot, n_way, q_queries,
                        distance, train):
    """Performs a single training episode for a Prototypical Network.
    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update
    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """

    losses = 0
    total_acc = 0
    criterion = CrossEntropyLoss().to(device)

    all_post = []
        
    # if train:
    #     model.train()
    # else:
    #     model.eval()

    for idx in range(x.shape[0]):

        if train:
            model.train()
        else:
            model.eval()

        optimiser.zero_grad()

        meta_batch = x[idx]
        # y_batch = y[idx]
        pid_batch = pid[idx][-n_way*q_queries:].unsqueeze(0)

        # print(meta_batch.shape)
        # print(y_batch)
        # print(pid[idx], pid_batch)

        s = meta_batch[:k_shot*n_way].view(-1, k_shot, 1, meta_batch.shape[-2], meta_batch.shape[-1]).float()
        q = meta_batch[k_shot*n_way:].view(-1, 1, meta_batch.shape[-2], meta_batch.shape[-1]).float()
        q_ssl = mask_for_query(s).to(device)
        s = s.view(1, -1, 1, meta_batch.shape[-2], meta_batch.shape[-1])
        q = q.unsqueeze(0)
        # print(s.shape, q.shape, q_ssl.shape)

        # slabel = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
        # qlabel = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
        slabel = torch.arange(n_way).repeat_interleave(k_shot).view(1, -1).to(device)
        qlabel = torch.arange(n_way).view(1, -1).to(device)

        slabel_oh = one_hot(slabel).float().to(device)
        qlabel_oh = one_hot(qlabel).float().to(device)

        ytest, cls_scores, _, _ = model(s, q, slabel_oh, qlabel_oh)
        ytest_ssl, cls_scores_ssl, _, _ = model(s, q_ssl, slabel_oh, qlabel_oh)
        # print(a1.shape, a2.shape)
        loss1 = criterion(cls_scores, qlabel.view(-1))
        loss2 = criterion(ytest, pid_batch.view(-1))
        loss3 = criterion(cls_scores_ssl, qlabel.view(-1))
        loss4 = criterion(ytest_ssl, pid_batch.view(-1))
        loss = loss1 + loss2 + 0.3*(loss3 + loss4)
        # loss = loss1 + loss2
        # print(f'Losses: Cls {loss1.item():.4f}, PID {loss2.item():.4f}, Cls_SSL {loss3.item():.4f}, PID_SSL {loss4.item():.4f}')
        # print(pid_batch.view(-1).cpu().numpy())

        model.eval()
        with torch.no_grad():
            val_cls_scores, _, _ = model(s, q, slabel_oh, qlabel_oh)
            _, val_preds = torch.max(val_cls_scores.view(1 * 5, -1).detach().cpu(), 1)
            acc = (torch.sum(val_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)
            # VAL_ACC.append(val_acc.item())
        
        if train:
            model.train()
        else:
            model.eval()

        if train:
            losses += loss
        else:
            losses += loss.item()
        total_acc += acc.item()

        all_post.append(acc.item())


    back_loss = losses / x.shape[0]
    post_acc = total_acc / x.shape[0]

    if train:
        back_loss.backward()
        optimiser.step()
        back_loss = back_loss.item()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    return back_loss, 0, post_acc, all_post

##############################################################################
# VARIABLE PROTONET EPISODE
##############################################################################
def proto_step_var(model, optimiser, loss_fn, x_support, x_query, q_num, y,
               device, n_way, k_shot, q_queries, distance):

    optimiser.zero_grad()
    # Sets up metric tracking
    total_query_loss = 0
    pre_acc_total = 0
    post_acc_total = 0

    all_post = []

    # Kepe track of the last used q_num access idx
    last_query_idx = 0
    # Iterate over num tasks
    for idx in range(x_support.shape[0]):

        x_task_train = x_support[idx]

        sub_q_num = q_num[idx*(n_way*q_queries): (idx+1)*(n_way*q_queries)]
        q_num_sub_sum = sum(sub_q_num)
        x_task_val = x_query[last_query_idx: (last_query_idx + q_num_sub_sum)]
        # Update tracking index
        last_query_idx += q_num_sub_sum

        # y value access is same as a fixed length batching
        y_task_train = y[idx][:(n_way * k_shot)]
        y_task_val = y[idx][(n_way * k_shot):]

        # print(x_task_train.shape, x_task_val.shape)
        # print(y_task_train, y_task_val)

        # We scale up the y task val to directly compare for loss but use majority for acc
        sub_q_nums_tens = torch.tensor(sub_q_num).to(device)
        scaled_up_query_y = torch.repeat_interleave(
            y_task_val, sub_q_nums_tens).to(device)

        # print(scaled_up_query_y, sub_q_nums_tens)

        unique_classes = torch.unique(scaled_up_query_y)
        sampled_indices = []

        for c in unique_classes:
            class_indices = (scaled_up_query_y == c).nonzero(as_tuple=True)[0]
            choice = class_indices[torch.randint(len(class_indices), (1,))].item()
            sampled_indices.append(choice)

        sampled_indices = torch.tensor(sampled_indices).to(device)
        # print(sampled_indices)

        s_v = x_task_train.view(1, -1, 1, x_task_train.shape[-2], x_task_train.shape[-1]).float()
        q_v = x_task_val[sampled_indices].unsqueeze(0).float()

        # slabel = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
        slabel = torch.arange(n_way).repeat_interleave(k_shot).view(1, -1).to(device)
        qlabel = scaled_up_query_y[sampled_indices].unsqueeze(0).to(device)

        slabel_oh = one_hot(slabel).float().to(device)
        qlabel_oh = one_hot(qlabel).float().to(device)
        # print(qlabel)

        # print(slabel, qlabel)
        # print(slabel_oh, qlabel_oh)
        model.eval()
        with torch.no_grad():
            val_cls_scores, _, _ = model(s_v, q_v, slabel_oh, qlabel_oh)
        # print(val_cls_scores.shape)

        _, val_preds = torch.max(val_cls_scores.view(1 * 5, -1).detach().cpu(), 1)
        post_acc = (torch.sum(val_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)

        # total_query_loss += query_loss.item()
        post_acc_total += post_acc.item()

        all_post.append(post_acc.item())

    # back_loss = total_query_loss / x_support.shape[0]
    avg_pre = pre_acc_total/x_support.shape[0]
    avg_post = post_acc_total/x_support.shape[0]

    return 0, avg_pre, avg_post, all_post

##############################################################################
# PAIRWISE DISTANCE CALCULATOR
##############################################################################
def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + sys.float_info.epsilon)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + sys.float_info.epsilon)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))

##############################################################################
# PROTOTYPE FUNCTION
##############################################################################
def compute_prototypes(support: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(n, k, -1).mean(dim=1)
    return class_prototypes

def proto_step_fixed_eval(device, model, optimiser, loss_fn, x, y, k_shot, n_way, q_queries,
                        distance, train):
    """Performs a single training episode for a Prototypical Network.
    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update
    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """

    losses = 0
    total_acc = 0
    criterion = CrossEntropyLoss().to(device)

    all_post = []
        
    # if train:
    #     model.train()
    # else:
    #     model.eval()

    for idx in range(x.shape[0]):

        # if train:
        #     model.train()
        # else:
        #     model.eval()
        model.eval()

        meta_batch = x[idx]
        # y_batch = y[idx]

        # print(meta_batch.shape)
        # print(y_batch)
        # print(pid[idx], pid_batch)

        s = meta_batch[:k_shot*n_way].view(-1, k_shot, 1, meta_batch.shape[-2], meta_batch.shape[-1]).float()
        q = meta_batch[k_shot*n_way:].view(-1, 1, meta_batch.shape[-2], meta_batch.shape[-1]).float()
        s = s.view(1, -1, 1, meta_batch.shape[-2], meta_batch.shape[-1])
        q = q.unsqueeze(0)
        # print(s.shape, q.shape, q_ssl.shape)

        # slabel = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
        # qlabel = torch.tensor([[0, 1, 2, 3, 4]]).to(device)
        slabel = torch.arange(n_way).repeat_interleave(k_shot).view(1, -1).to(device)
        qlabel = torch.arange(n_way).view(1, -1).to(device)

        slabel_oh = one_hot(slabel).float().to(device)
        qlabel_oh = one_hot(qlabel).float().to(device)

        model.eval()
        with torch.no_grad():
            val_cls_scores, _, _ = model(s, q, slabel_oh, qlabel_oh)
            _, val_preds = torch.max(val_cls_scores.view(1 * 5, -1).detach().cpu(), 1)
            acc = (torch.sum(val_preds == qlabel.detach().cpu()).float()) / qlabel.size(1)
            # VAL_ACC.append(val_acc.item())

        total_acc += acc.item()

        all_post.append(acc.item())

    post_acc = total_acc / x.shape[0]

    return 0, 0, post_acc, all_post

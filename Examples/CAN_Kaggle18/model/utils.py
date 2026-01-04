import os
import pandas as pd
import torch
import librosa
import numpy as np
import random
import scipy

def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot


def episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['target'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + n_query)
        # if k_shot == 1:
        #     support = data[indicies[0]][0]
        # else:
        support = torch.stack([data[i][0] for i in indicies[:-n_query]])
        if n_query == 1:
            query = data[indicies[-1]][0]
        else:
            query = torch.stack([data[i][0] for i in indicies[-n_query:]])
        support_set.append(support)
        query_set.append(query)

    return torch.stack(support_set), torch.stack(query_set), torch.tensor(ways).unsqueeze(0)

def wav_episodic_sampling(c_way, k_shot, base_classes, data, n_query=1):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    wav_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['target'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + n_query)

        support = torch.stack([data[i][0] for i in indicies[:-n_query]])
        wav = torch.stack([data[i][2] for i in indicies[:-n_query]])

        if n_query == 1:
            query = data[indicies[-1]][0]
        else:
            query = torch.stack([data[i][0] for i in indicies[-n_query:]])

        support_set.append(support)
        query_set.append(query)
        wav_set.append(wav)

    return torch.stack(support_set), torch.stack(query_set), torch.tensor(ways).unsqueeze(0), torch.stack(wav_set)


import torch
import torch.nn.functional as F

def feature_mask(time_mask: torch.Tensor, total_stride: int = 16) -> torch.Tensor:
    """
    Maps a 1D input time mask (T_input=160) to a 1D feature level mask (T_feature=10).

    Args:
        time_mask (torch.Tensor): The 1D input mask (T_input) or 2D (B, T_input). 
                                  Assumes T_input is 160.
        total_stride (int): The total downsampling factor on the time axis (16).

    Returns:
        torch.Tensor: The 1D feature mask (T_feature) or 2D (B, T_feature), with T_feature=10.
    """
    
    # --- 1. Input Validation and Reshaping ---
    if time_mask.dim() == 1:
        # Reshape to (1, 1, T_input) for F.max_pool1d: (B, C, L) where C=1
        mask_1d_r = time_mask.unsqueeze(0).unsqueeze(0)  
    elif time_mask.dim() == 2:
        # Reshape to (B, 1, T_input)
        mask_1d_r = time_mask.unsqueeze(1) 
    else:
        raise ValueError("Input mask must be 1D (T) or 2D (B, T).")
        
    mask_1d_r = mask_1d_r.float()
    
    T_input = mask_1d_r.shape[-1]
    
    # Check for expected time dimension size (160)
    if T_input != 160:
         # For robustness, handle cases where T_input is not 160 or not divisible by 16
         padding_needed = (total_stride - (T_input % total_stride)) % total_stride
         mask_1d_r = F.pad(mask_1d_r, (0, padding_needed), 'constant', 0)
    
    # --- 2. Apply Time Downsampling (R=16) ---
    # Max pooling ensures if *any* of the 16 original time steps was unmasked (1), 
    # the pooled feature token remains unmasked (1).
    feature_mask_pooled = F.max_pool1d(
        mask_1d_r, 
        kernel_size=total_stride, 
        stride=total_stride
    ) # Shape: (B, 1, T_feature=10)

    # --- 3. Convert back to Binary and Output Format ---
    # Squeeze out the channel dimension (C=1) and convert to binary (0 or 1) integer tensor
    feature_mask = feature_mask_pooled.squeeze(1).ceil().long()

    # Return correct shape based on input (1D input should return 1D output)
    if time_mask.dim() == 1:
        return feature_mask.squeeze(0) # Shape: (T_feature=10)
    
    return feature_mask

def mask_for_query(support):
    """
    support: [Nway=5, Nshot=5, 1, 128, T]
    
    Returns:
        ssl_queries: [1, 5, 1, 128, T]
    """

    Nway, Nshot, C, F, T = support.shape
    queries = []

    for c in range(Nway):

        # pick one support sample randomly from this class
        idx = torch.randint(0, Nshot, (1,)).item()
        x = support[c, idx]    # [1, 128, T]
        # x = support[c, 0]    # [1, 128, T]

        # compute time-wise energy
        energy = x.pow(2).mean(dim=1).squeeze(0)  # [T]

        # select top-50% energy frames
        sorted_idx = torch.argsort(energy, descending=True)
        keep_len = int(0.6 * T)
        keep_idx = sorted_idx[:keep_len]

        # binary mask over time frames
        mask = torch.zeros(T, dtype=torch.bool, device=x.device)
        mask[keep_idx] = True

        # apply mask: keep top-energy frames
        x_masked = x.clone()
        
        # x_masked[:, :, ~mask] = 0   # zero out low-energy frames

        noise_scale = 0.3 * x.std().clamp(min=1e-8) 
        # noise_mean  = 0.3 * x.mean().clamp(min=0.0)
        noise = torch.abs(torch.randn_like(x)) * noise_scale
        x_masked[:, :, ~mask] = noise[:, :, ~mask]

        # reshape to [1,128,T] → [1,1,128,T] for query
        x_masked = x_masked.unsqueeze(0)  # [1,1,128,T]
        queries.append(x_masked)

    # stack: becomes [5,1,1,128,T]
    queries = torch.cat(queries, dim=0)

    # add batch dim → [1,5,1,128,T]
    queries = queries.unsqueeze(0)

    return queries


def mask_for_query_and_att(support):
    """
    support: [Nway=5, Nshot=5, 1, 128, T]
    
    Returns:
        ssl_queries: [1, 5, 1, 128, T]
    """

    Nway, Nshot, C, F, T = support.shape
    queries = []
    feature_masks = []

    for c in range(Nway):

        # pick one support sample randomly from this class
        idx = torch.randint(0, Nshot, (1,)).item()
        x = support[c, idx]    # [1, 128, T]
        # x = support[c, 0]    # [1, 128, T]

        # compute time-wise energy
        energy = x.pow(2).mean(dim=1).squeeze(0)  # [T]

        # select top-50% energy frames
        sorted_idx = torch.argsort(energy, descending=True)
        keep_len = int(0.6 * T)
        keep_idx = sorted_idx[:keep_len]

        # binary mask over time frames
        mask = torch.zeros(T, dtype=torch.bool, device=x.device)
        mask[keep_idx] = True

        f_mask = feature_mask(mask, total_stride=16)  # [T_feature=10]
        # print(f_mask.shape)

        # apply mask: keep top-energy frames
        x_masked = x.clone()
        
        # x_masked[:, :, ~mask] = 0   # zero out low-energy frames

        noise_scale = 0.3 * x.std().clamp(min=1e-8) 
        # noise_mean  = 0.3 * x.mean().clamp(min=0.0)
        noise = torch.abs(torch.randn_like(x)) * noise_scale
        x_masked[:, :, ~mask] = noise[:, :, ~mask]

        # reshape to [1,128,T] → [1,1,128,T] for query
        x_masked = x_masked.unsqueeze(0)  # [1,1,128,T]
        queries.append(x_masked)
        feature_masks.append(f_mask.unsqueeze(0)) 

    # stack: becomes [5,1,1,128,T]
    queries = torch.cat(queries, dim=0)
    feature_masks = torch.cat(feature_masks, dim=0)  # [5, T_feature=10]

    # add batch dim → [1,5,1,128,T]
    queries = queries.unsqueeze(0)

    return queries, feature_masks


import matplotlib.pyplot as plt

def plot_mel_db(mel_db, cmap="magma"):
    if hasattr(mel_db, "detach"):
        mel_db = mel_db.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label="dB")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bins")
    plt.title("Log-Mel Spectrogram (dB)")
    plt.tight_layout()
    plt.show()

import torch
import torch.nn.functional as F

def attention_constraint(
    attn_map_model: torch.Tensor, 
    feature_mask_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the selective attention loss for weights in the range (1, 2) 
    (due to residual connection), penalizing weights towards a target of 1.0 
    only at the masked indices (where feature_mask_gt == 0).

    Args:
        attn_map_model (torch.Tensor): The cross-attention map (B, N_Q, N_P, T).
        feature_mask_gt (torch.Tensor): The ground truth binary mask (N_Q, T).
        lambda_attn (float): Weighting hyperparameter for the loss.

    Returns:
        torch.Tensor: The calculated selective attention loss.
    """
    
    # 1. Define Penalty Target (T_penalty)
    # The target for weights corresponding to masked (noisy) features is 1.0.
    T_penalty = 1.0
    
    # 2. Prepare the Penalty Mask (P)
    B, N_Q, N_P, T = attn_map_model.shape
    
    # Invert the mask: 0 -> 1 (penalize), 1 -> 0 (ignore)
    # Shape after (1 - ...) and unsqueeze: (N_Q, 1, T)
    penalty_mask = (1.0 - feature_mask_gt.float()).unsqueeze(1) 
    
    # Expand to (N_Q, N_P, T) by broadcasting across the N_P dimension
    penalty_mask_expanded = penalty_mask.expand(-1, N_P, -1) 
    
    # 3. Calculate the Weighted Squared Error against Target 1.0
    
    # MSE loss: (A_model - T_penalty)^2
    squared_error = (attn_map_model - T_penalty).pow(2).squeeze(0)

    # print(penalty_mask_expanded)
    
    # Element-wise multiplication applies the penalty mask
    penalized_error = squared_error * penalty_mask_expanded 

    # print(penalized_error.shape)
    
    # 4. Final Loss Calculation
    loss = penalized_error.sum()
    
    return loss
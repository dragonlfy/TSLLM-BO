# -*- coding: utf-8 -*-
"""Time‑LLM with PPO Reinforcement Learning Training Script
========================================
特点
------
* 采用PPO算法进行强化学习训练
* 策略网络：Time-LLM模型作为策略网络，输出预测序列
* 价值网络：估计状态价值，辅助策略优化
* 奖励函数：结合预测精度与形态相似度设计
* 保留原有时间序列预测的评估方式
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from models.softdtw import SoftDTW
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from torch.optim import AdamW, lr_scheduler
from tqdm.auto import tqdm

from data_provider.data_factory import data_provider
from models import Autoformer, DLinear, TimeLLM, TRLLM, TRLLM_T
from utils.tools import EarlyStopping, del_files, vali


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# ----------------------------- CLI ----------------------------- #
parser = argparse.ArgumentParser("Time-LLM RL Training with PPO")

# 原有参数保留
parser.add_argument('--task_name',      type=str, default='long_term_forecast',
                    choices=['long_term_forecast', 'short_term_forecast',
                             'imputation', 'classification', 'anomaly_detection'])
parser.add_argument('--is_training',    type=int, default=1)
parser.add_argument('--model_id',       type=str, default='test')
parser.add_argument('--model_comment',  type=str, default='none')
parser.add_argument('--model',          type=str, default='TimeLLM',
                    choices=['Autoformer', 'DLinear', 'TimeLLM', 'TRLLM', 'TRLLM_T'])
parser.add_argument('--seed',           type=int, default=2021)

parser.add_argument('--data',       type=str, default='ETTm1')
parser.add_argument('--root_path',  type=str, default='./dataset')
parser.add_argument('--data_path',  type=str, default='ETTh1.csv')
parser.add_argument('--features',   type=str, default='M', choices=['M', 'S', 'MS'])
parser.add_argument('--target',     type=str, default='OT')
parser.add_argument('--loader',     type=str, default='modal')
parser.add_argument('--freq',       type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--seq_len',  type=int, default=96)
parser.add_argument('--label_len',type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

parser.add_argument('--enc_in',   type=int, default=7)
parser.add_argument('--dec_in',   type=int, default=7)
parser.add_argument('--c_out',    type=int, default=7)
parser.add_argument('--d_model',  type=int, default=16)
parser.add_argument('--n_heads',  type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff',     type=int, default=32)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor',   type=int, default=1)
parser.add_argument('--dropout',  type=float, default=0.1)
parser.add_argument('--embed',    type=str, default='timeF', choices=['timeF', 'fixed', 'learned'])
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride',   type=int, default=8)
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--llm_model', type=str, default='LLAMA', choices=['LLAMA', 'GPT2', 'BERT'])
parser.add_argument('--llm_dim',   type=int, default=4096)
parser.add_argument('--llm_layers',type=int, default=6)

parser.add_argument('--itr',          type=int, default=1)
parser.add_argument('--des',          type=str, default='test')
parser.add_argument('--percent',      type=int, default=100)
parser.add_argument('--num_workers',  type=int, default=64)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--align_epochs', type=int, default=10)
parser.add_argument('--batch_size',   type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--patience',     type=int, default=10)
parser.add_argument('--learning_rate',type=float, default=3e-5)  # RL通常学习率更小
parser.add_argument('--loss',         type=str, default='MSE')
parser.add_argument('--lradj',        type=str, default='type1')
parser.add_argument('--pct_start',    type=float, default=0.2)
parser.add_argument('--use_amp',      action='store_true')

# 添加RL相关参数
parser.add_argument('--rl_algorithm',  type=str, default='ppo', choices=['ppo', 'reinforce'])
parser.add_argument('--gamma',         type=float, default=0.99)  # 折扣因子
parser.add_argument('--gae_lambda',    type=float, default=0.95)  # GAE参数
parser.add_argument('--ppo_clip',      type=float, default=0.2)   # PPO剪辑参数
parser.add_argument('--value_coef',    type=float, default=0.5)   # 价值损失系数
parser.add_argument('--entropy_coef',  type=float, default=0.01)  # 熵奖励系数
parser.add_argument('--vf_clip_param', type=float, default=1.0)   # 价值函数剪辑
parser.add_argument('--reward_mse_weight', type=float, default=1.0)
parser.add_argument('--reward_wave_weight', type=float, default=0.2)
parser.add_argument('--reward_dtw_weight', type=float, default=0.1)
parser.add_argument('--reward_emd_weight', type=float, default=0.1)

args = parser.parse_args()

# ------------------------ Repro / Env ------------------------- #
np.random.seed(args.seed)
torch.manual_seed(args.seed)

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
# plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(
    kwargs_handlers=[ddp],
    # deepspeed_plugin=plugin,
    mixed_precision='fp16' if args.use_amp else 'no',
    device_placement=True,
)

device = accelerator.device

# --------------------------- Data ----------------------------- #
train_data, train_loader = data_provider(args, 'train')
val_data,   val_loader   = data_provider(args, 'val')
test_data,  test_loader  = data_provider(args, 'test')

# -------------------------- Model ---------------------------- #
# 策略网络：原Time-LLM模型
model_cls = dict(Autoformer=Autoformer.Model, DLinear=DLinear.Model,
                 TimeLLM=TimeLLM.Model, TRLLM=TRLLM.Model, TRLLM_T=TRLLM_T.Model)[args.model]
policy_net = model_cls(args).float()


# 价值网络：估计状态价值
class ValueNetwork(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim=128):  # 新增input_dim参数
        super().__init__()
        # 输入特征数 = seq_len * input_dim（匹配实际输入[B, seq_len, input_dim]）
        self.fc1 = nn.Linear(seq_len * input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出状态价值
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: [B, seq_len, input_dim]（实际输入形状）
        B, L, D = x.shape
        x = x.reshape(B, -1)  # [B, L*D]，例如[B, 96*7=672]
        x = self.activation(self.fc1(x))  # 现在维度匹配（672x128）
        x = self.activation(self.fc2(x))
        return self.fc3(x)

value_net = ValueNetwork(seq_len=args.seq_len, input_dim=args.enc_in).float()

# 优化器
policy_optimizer = AdamW(policy_net.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
value_optimizer = AdamW(value_net.parameters(), lr=args.learning_rate * 2, betas=(0.9, 0.95))  # 价值网络学习率可更高

# 准备加速器
policy_net, value_net, policy_optimizer, value_optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
    policy_net, value_net, policy_optimizer, value_optimizer, train_loader, val_loader, test_loader)

if args.use_amp:
    policy_net = policy_net.to(torch.bfloat16)
    value_net = value_net.to(torch.bfloat16)
criterion = nn.MSELoss()
soft_dtw = SoftDTW(gamma=0.1, normalize=True)

stopper = EarlyStopping(accelerator, patience=args.patience)

# ---------------------- RL核心组件 ----------------------- #

def compute_reward(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """计算奖励 [B]"""
    B, L, C = pred.shape
    
    # 1. MSE奖励（误差小则奖励高）
    mse = torch.mean((pred - tgt)**2, dim=(1, 2))  # [B]
    reward_mse = -args.reward_mse_weight * mse  # 负MSE作为奖励
    
    # 2. 小波相似度奖励（相似度高则奖励高）
    wave_sim = wavelet_similarity(pred, tgt)  # [B]
    reward_wave = args.reward_wave_weight * wave_sim
    
    # 3. DTW奖励（DTW距离小则奖励高）
    dtw_dist = soft_dtw(pred, tgt).mean(dim=1)  # [B]
    reward_dtw = -args.reward_dtw_weight * dtw_dist  # 负距离作为奖励
    
    # 4. EMD奖励
    emd_dist = emd_1d(pred, tgt)  # [B]
    reward_emd = -args.reward_emd_weight * emd_dist
    
    # 总奖励
    total_reward = reward_mse + reward_wave
    return total_reward

def wavelet_similarity(x: torch.Tensor, y: torch.Tensor, wave: str = 'haar') -> torch.Tensor:
    """小波相似度 [B]"""
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    B = x.shape[0]
    
    coeffs_x = pywt.wavedec(x_np, wavelet=wave, axis=1, level=3)
    coeffs_y = pywt.wavedec(y_np, wavelet=wave, axis=1, level=3)
    
    sim = 0.0
    for cx, cy in zip(coeffs_x, coeffs_y):
        cx = torch.as_tensor(cx, device=x.device, dtype=torch.float32)
        cy = torch.as_tensor(cy, device=x.device, dtype=torch.float32)
        sim += F.cosine_similarity(
            cx.reshape(B, -1), 
            cy.reshape(B, -1), 
            dim=1
        )
    
    return sim / len(coeffs_x)  # 平均各层相似度

def emd_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """EMD距离 [B]"""
    cdf_a = a.cumsum(dim=1)
    cdf_b = b.cumsum(dim=1)
    return (cdf_a - cdf_b).abs().mean(dim=[1, 2])

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor, dones=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    广义优势估计 (GAE)
    rewards: [B], values: [B], next_value: [B]
    返回: 优势 [B], 目标价值 [B]
    """
    if dones is None:
        dones = torch.zeros_like(rewards)
    
    # 计算TD误差
    delta = rewards + args.gamma * next_value * (1 - dones) - values
    # 计算优势
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(delta))):  # 这里batch内视为独立时间步
        gae = delta[t] + args.gamma * args.gae_lambda * gae * (1 - dones[t])
        advantages[t] = gae
    
    # 目标价值 = 优势 + 价值估计
    target_values = advantages + values
    return advantages, target_values


def ppo_update(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, advantages: torch.Tensor, 
              values: torch.Tensor, target_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PPO更新步骤"""
    # 策略损失
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))
    
    # 价值损失（带剪辑）
    vf_loss1 = criterion(values, target_values)
    values_clipped = values + torch.clamp(values - values.detach(), -args.vf_clip_param, args.vf_clip_param)
    vf_loss2 = criterion(values_clipped, target_values)
    value_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))
    
    # 总损失
    total_loss = policy_loss + args.value_coef * value_loss
    return total_loss, policy_loss, value_loss

def get_model_dtype(model):
    """获取模型参数的数据类型（处理DDP包装的情况）"""
    # 如果模型被DDP包装，取内部模型
    if hasattr(model, 'module'):
        model = model.module
    # 取模型的第一个参数的dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        # 模型没有参数的极端情况（通常不会发生）
        return torch.float32

# 提前获取模型数据类型
model_dtype = get_model_dtype(policy_net)

# ------------------------- 训练循环 -------------------------- #
for epoch in range(args.train_epochs):
    policy_net.train()
    value_net.train()
    
    epoch_policy_loss = []
    epoch_value_loss = []
    epoch_total_reward = []
    
    with tqdm(train_loader, disable=not accelerator.is_local_main_process) as bar:
        for step, (bx, by, bx_m, by_m) in enumerate(bar):
            bx, by = bx.float().to(device), by.float().to(device)
            bx_m, by_m = bx_m.float().to(device), by_m.float().to(device)
            B = bx.shape[0]
            
            # 1. 收集轨迹（旧策略采样）
            with torch.no_grad():
                # 旧策略生成预测（添加高斯噪声实现探索）
                dec_inp = torch.cat([by[:, :args.label_len], torch.zeros_like(by[:, -args.pred_len:])], dim=1).float()
                old_out = policy_net(bx, bx_m, dec_inp, by_m)[:, -args.pred_len:, :]
                tgt = by[:, -args.pred_len:, :]
                
                # 计算状态价值
                values = value_net(bx).squeeze(1)  # [B]
                
                # 计算奖励
                rewards = compute_reward(old_out, tgt)  # [B]
                epoch_total_reward.append(rewards.mean().item())
                
                # 计算下一个状态价值（这里简化：使用当前价值网络估计）
                next_values = value_net(bx).squeeze(1)  # 简化处理，实际应使用下一个状态
                
                # 计算优势和目标价值
                advantages, target_values = compute_gae(rewards, values, next_values)
                
                # 旧策略的log概率（对于确定性策略，这里用MSE的负对数作为近似）
                # 注：确定性策略的概率处理是近似，也可添加高斯噪声转为随机策略
                old_mse = torch.mean((old_out - tgt)**2, dim=(1,2))
                old_log_probs = -0.5 * old_mse  # 近似log概率

            # 2. 新策略采样
            dec_inp_new = torch.cat([by[:, :args.label_len], torch.zeros_like(by[:, -args.pred_len:])], dim=1).float()
            new_out = policy_net(bx, bx_m, dec_inp_new, by_m)[:, -args.pred_len:, :]
            
            # 新策略的log概率
            new_mse = torch.mean((new_out - tgt)** 2, dim=(1,2))
            new_log_probs = -0.5 * new_mse

            # 3. PPO更新
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            # 计算损失
            total_loss, policy_loss, value_loss = ppo_update(
                old_log_probs.detach(), new_log_probs, advantages.detach(),
                value_net(bx).squeeze(1), target_values.detach()
            )
            
            # 反向传播
            accelerator.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            
            policy_optimizer.step()
            value_optimizer.step()
            
            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())
            
            bar.set_postfix({
                "pol_loss": np.mean(epoch_policy_loss),
                "val_loss": np.mean(epoch_value_loss),
                "reward": np.mean(epoch_total_reward)
            })

    # ---------- Evaluation ----------
    # 用监督损失评估性能（保持原有评估方式）
    val_loss, val_mae = vali(args, accelerator, policy_net, val_data, val_loader, criterion, nn.L1Loss())
    test_loss, test_mae = vali(args, accelerator, policy_net, test_data, test_loader, criterion, nn.L1Loss())
    accelerator.print(
        f"[Epoch {epoch+1}] "
        f"pol_loss: {np.mean(epoch_policy_loss):.4f} | "
        f"val_loss: {np.mean(epoch_value_loss):.4f} | "
        f"reward: {np.mean(epoch_total_reward):.4f} | "
        f"val_mae: {val_mae:.3f} | "
        f"test_mae: {test_mae:.3f}"
    )

    # ---------- CSV Logging ----------
    if accelerator.is_local_main_process:
        row_dict = dict(args.__dict__)
        row_dict['current_epoch'] = epoch + 1
        row_dict['policy_loss'] = np.mean(epoch_policy_loss)
        row_dict['value_loss'] = np.mean(epoch_value_loss)
        row_dict['avg_reward'] = np.mean(epoch_total_reward)
        row_dict['vali_mse'] = val_loss
        row_dict['vali_mae'] = val_mae
        row_dict['test_mse'] = test_loss
        row_dict['test_mae'] = test_mae
        row_dict['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        csv_filename = f"result_rl_{args.model}_train.csv"
        fieldnames = list(row_dict.keys())
        file_exists = os.path.exists(csv_filename)
        
        try:
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_dict)
        except Exception as e:
            accelerator.print(f"CSV写入错误: {e}")

        stopper(val_loss, policy_net, 'checkpoints/rl_model.ckpt')
        if stopper.early_stop:
            accelerator.print('早停触发')
            break

# ------------------------- 清理 --------------------------- #
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    del_files('./checkpoints')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
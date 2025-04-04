# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import kornia.augmentation as aug
import torch.nn as nn
from model import DQN

random_shift = nn.Sequential(aug.RandomCrop((80, 80)), nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
aug = random_shift

class Agent():
  '''
  代理器：包括环境和模型
  '''
  def __init__(self, args, env):
    self.args = args
    self.action_space = env.action_space()
    self.atoms = args.atoms # 分布dqn
    self.Vmin = args.V_min # 分布dqn
    self.Vmax = args.V_max # 分布dqn
    '''
    support应该是模拟真实环境下不同选择对应不同的结果分布
    比如选择开车，分为堵车和不堵车的耗时
    选择地铁，基本上时间是不堵车，耗时比堵车短比不堵车长
    那么如何权衡一个动作的在不同场景下的选择呢？
    比如如果必须要赶时间去坐到火车，那么就应该选择一个一定不会迟到的动作
    通过分布加权来选择一个动作，可知选择开车的动作平均耗时会比选择地铁的动作要长，所以这时候就应该要选择地铁的动作
    '''
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1) # 分布dqn
    self.batch_size = args.batch_size # 批量大小
    self.n = args.multi_step # n步分布dqn
    self.discount = args.discount # 折扣因子
    self.norm_clip = args.norm_clip
    self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1. # 系数 todo 作用

    # 构建了两个相同的网络，todo 作用是什么？
    # online_net是正常的dqn网络
    # momentum_net是一个动量网络，和online_net的参数是一样的，属于CURL的对比学习
    self.online_net = DQN(args, self.action_space).to(device=args.device)
    self.momentum_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      # 加载预训练模型，这里可能是原作者自己的关系需要实现这种映射新的键值名
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()
    # 将 online_net的参数同步给 momentum_net
    # 在initialize_momentum_net方法里面讲momentum_net参数设置为requires_grad=False，但是外面又设置为momentum_net为true，这两个操作互相矛盾吧
    # 答：它们并不冲突。调用momentum_net.train()只是让网络的BN或Dropout等层以“训练”模式工作，而param_k.requires_grad=False则确保它的权重不通过反向传播更新。这样可以在保持某些层“训练”行为的同时，通过手动方式（而非梯度）来更新这些参数。
    # **Answer**  在训练模式下，BatchNorm 会使用当前批次的数据来计算均值和方差并更新运行统计量，Dropout 会随机屏蔽部分神经元以防止过拟合。切换到评估模式后，BatchNorm 使用之前保存的均值和方差，Dropout 不再随机屏蔽任何神经元，从而保证输出结果的稳定性。
    self.initialize_momentum_net()
    self.momentum_net.train() # 从这里来看，momentum_net应该算是一个目标网络TargetNet吧？todo

    # 这里创建了一个目标网络，那么momentum_net不是目标网络吗？momentum_net的作用是什么？todo
    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net() # 同步online_net到target_net
    self.target_net.train() # 设置为训练模式 todo targetNet是训练？todo和我的代码不一样
    # targetnet同样也是设置为训练模式不参与梯度计算
    for param in self.target_net.parameters():
      param.requires_grad = False

    # 再次确认momentum_net不参与梯度计算
    for param in self.momentum_net.parameters():
      param.requires_grad = False
    # 优化器仅优化onlone_net
    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    '''
    重置噪声，增加探索
    重置噪声仅对online_net有效
    '''
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    # 预测动作
    # 由于使用了分布DQN，所以这里的动作是一个分布
    with torch.no_grad():
      a, _ = self.online_net(state.unsqueeze(0))
      # 加权求和，得到每个动作的Q值，选择最大Q值的动作
      return (a * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    # 根据ε-greedy策略选择动作
    # 这里的epsilon是一个小值，表示选择随机动作的概率
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    '''
    训练模型
    '''
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    # 对states进行数据增强，即随机裁剪、填充、再随机裁剪
    # 这里生产了两个大体相同但是又有差异的状态，应该是分别传入到online_net和momentum_net中
    # 这里是需要特意产生的，后续有用
    aug_states_1 = aug(states).to(device=self.args.device)
    aug_states_2 = aug(states).to(device=self.args.device)
    # Calculate current state probabilities (online network noise already sampled)
    # 先用原始的观察生成动作Q值分布
    log_ps, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    # 在用online_net的输出对比学习的特征
    _, z_anch = self.online_net(aug_states_1, log=True)
    # 使用momentum_net的输出对比学习的特征
    _, z_target = self.momentum_net(aug_states_2, log=True)
    # 将最比学习的特征投射到一个新的特征空间，可以得到更鲁棒的特征表示
    # todo 是否也表示可以不进行映射？
    z_proj = torch.matmul(self.online_net.W, z_target.T)
    # 然后与锚点特征 z_anch 计算相似性（通过矩阵乘法）
    '''
    作用: 计算锚点特征 z_anch 和目标特征 z_proj 的相似性。
    z_anch 是从 aug_states_1（增强后的状态）中提取的特征。
    z_proj 是从 aug_states_2（增强后的状态）中提取的特征，并通过 W 投影到新的特征空间。
    矩阵乘法计算了 z_anch 和 z_proj 之间的点积，表示它们在特征空间中的相似性。
    CURL中的作用:
    点积相似性用于对比学习的损失计算，目标是让正样本对（来自同一状态的增强视图）具有更高的相似性。
    '''
    logits = torch.matmul(z_anch, z_proj)
    '''
    作用: 对 logits 进行数值稳定性处理。
    torch.max(logits, 1)[0] 计算每一行的最大值。
    [:, None] 将最大值扩展为列向量，方便与 logits 的每一行相减。
    通过减去最大值，避免在后续计算 softmax 时出现数值溢出问题。
    CURL中的作用:
    确保对比学习的数值计算稳定，避免梯度爆炸或溢出
    '''
    logits = (logits - torch.max(logits, 1)[0][:, None])
    '''
    作用: 缩放 logits 的值。
    乘以一个小的缩放因子（0.1），使得 logits 的范围更小。
    这会影响 softmax 的输出分布，使其更加平滑。
    '''
    logits = logits * 0.1
    '''
    作用: 生成对比学习的目标标签。
    torch.arange(logits.shape[0]) 生成从 0 到 batch_size-1 的整数序列。
    每个样本的标签是其对应的索引，表示正样本对的匹配关系。
    '''
    labels = torch.arange(logits.shape[0]).long().to(device=self.args.device)
    '''
    moco_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.device)
    作用: 计算对比学习的损失。
    使用交叉熵损失（CrossEntropyLoss）来优化 logits 和目标标签之间的匹配。
    正样本对的相似性被最大化，负样本对的相似性被最小化。
    CURL中的作用:
    对比学习的核心目标是通过优化交叉熵损失，让模型学习到更鲁棒的状态表示。
    这种表示能够捕获状态的本质特征，从而提升强化学习的性能。

    
    这是对比学习中常用的一种“行对列”匹配方式：  
- 每个样本在 batch 中都有唯一的“正确匹配”（即自身）。  
- 因此，行 i 的正确标签就是整数 i。  
- 用交叉熵损失来要求“行 i 与列 i”这一对的相似度最高，其他列为负样本。  

通过将标签设为 0 到 batch_size-1，保证第 i 行只在第 i 列处具有最高分，使得同一增强视图的样本两两匹配，形成正样本对。

todo 后续模型运行时查看其的size和尺寸
    '''
    moco_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.device)

    # 获取实际执行动作的概率分布
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      # 先用训练的online_net来预测下一个状态的动作分布选择最大Q值的动作
      pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      # 然后用目标网络来预测下一个状态的Q值动作分布
      self.target_net.reset_noise()  # Sample new target net noise
      pns, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      # 在用online_net选择的动作来选择下一个状态的Q值，这边时double dqn的计算公式
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      # todo 对比这里的n步dqnde的未来折扣值和我的计算有什么区别
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # 这边实在计算分布dqn吧
      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    # 计算分布dqn的kl损失
    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    # 综合分布dqn的损失+对比损失
    loss = loss + (moco_loss * self.coeff)
    # 计算梯度，更新模型
    self.online_net.zero_grad()
    curl_loss = (weights * loss).mean()
    curl_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    # 缓冲区更新权重优先级
    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    '''
    更新目标网络，权重是1，全更新
    '''
    self.target_net.load_state_dict(self.online_net.state_dict())

  def initialize_momentum_net(self):
    for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
      param_k.data.copy_(param_q.data) # update
      param_k.requires_grad = False  # not update by gradient

  # Code for this function from https://github.com/facebookresearch/moco
  @torch.no_grad()
  def update_momentum_net(self, momentum=0.999):
    for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
      param_k.data.copy_(momentum * param_k.data + (1.- momentum) * param_q.data) # update

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  # 计算动作的Q值
  # 这里的Q值是分布dqn的Q值，直接将动作和分布相乘，然后求和得到q值，也就是每个动作的q值分布在求加权和
  def evaluate_q(self, state):
    with torch.no_grad():
      a, _ = self.online_net(state.unsqueeze(0))
      return (a * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

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
import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.module_name = 'noisy_linear'
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    '''
    缩放
    缩放到[-1, 1]之间

    这行代码是在进行噪声缩放操作，具体实现了 Factorised Noisy Networks 中的特殊噪声生成方法。让我详细解释：

  ```python
  x.sign().mul_(x.abs().sqrt_())
  ```

  这行代码可以分解为以下步骤：

  1. `x.sign()` - 获取输入张量 x 的符号（+1 或 -1）
  2. `x.abs()` - 计算输入张量的绝对值
  3. `sqrt_()` - 计算绝对值的平方根
  4. `mul_()` - 将符号与平方根相乘（原位操作）

  ### 作用
  - 这个操作实现了论文 [*Noisy Networks for Exploration*](https://arxiv.org/abs/1706.10295) 中提出的噪声生成方法
  - 生成的噪声遵循特定分布，比普通的高斯噪声更适合探索
  - 输出范围在 [-1, 1] 之间，但分布不是均匀的

  ### 数学表达式
  如果输入是随机变量 ε，输出 f(ε) 的计算公式为：
  ```
  f(ε) = sign(ε) * sqrt(|ε|)
  ```

  这种噪声生成方法有助于：
  - 提供更好的探索性能
  - 保持噪声的有界性
  - 产生适当的探索-利用平衡

  在 NoisyLinear 层中，这个噪声用于权重和偏置的扰动，帮助网络进行更有效的探索。
    '''
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    '''
    重制噪声层

    所有的权重的偏置重置到-1到1之间
    '''
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  '''
  DQN网络
  '''
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms # 离散动作的分布
    self.action_space = action_space # 动作数

    # 选择两个不同的网络结构
    # canonical 架构: 更深的网络，适合需要高精度特征提取的任务，但计算量较大。
    # data-efficient 架构: 更浅的网络，适合资源受限或需要快速训练的场景，但特征提取能力可能稍弱。
    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std) # 状态的价值分布
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std) # 动作的优势分布

    # 对比学习的头，todo作用
    self.W_h = nn.Parameter(torch.rand(self.conv_output_size, args.hidden_size))
    self.W_c = nn.Parameter(torch.rand(args.hidden_size, 128))
    self.b_h = nn.Parameter(torch.zeros(args.hidden_size))
    self.b_c = nn.Parameter(torch.zeros(128))
    '''
    在代码中，`W` 是对比学习模块的一部分，用于生成对比学习的特征表示。具体来说，它是一个可学习的参数矩阵，用于对特征进行投影和变换。

    ---

    ### 代码片段
    ```python
    self.W_h = nn.Parameter(torch.rand(self.conv_output_size, args.hidden_size))
    self.W_c = nn.Parameter(torch.rand(args.hidden_size, 128))
    self.b_h = nn.Parameter(torch.zeros(args.hidden_size))
    self.b_c = nn.Parameter(torch.zeros(128))
    self.W = nn.Parameter(torch.rand(128, 128))
    ```

    ---

    ### `W` 的作用
    1. **对比学习中的特征投影**:
      - `W` 是一个可学习的参数矩阵，用于将对比学习的特征表示投影到一个新的特征空间。
      - 在对比学习中，特征投影可以帮助模型学习到更鲁棒的特征表示，从而更好地进行对比损失的计算。

    2. **计算对比学习的相似性**:
      - 在 `forward` 方法中，`W` 被用来计算对比学习特征之间的相似性：
        ```python
        z_proj = torch.matmul(self.online_net.W, z_target.T)
        logits = torch.matmul(z_anch, z_proj)
        ```
      - 这里，`W` 用于将目标特征 `z_target` 投影到一个新的空间，然后与锚点特征 `z_anch` 计算相似性（通过矩阵乘法）。

    3. **对比学习的目标**:
      - 通过 `W` 的投影，模型可以学习到更好的特征表示，使得相似的状态在特征空间中更接近，不相似的状态更远离。
      - 这对于 CURL（对比学习）非常重要，因为它依赖于特征的相似性来优化对比损失。

    ---

    ### 总结
    `W` 是对比学习模块中的一个核心参数，用于特征投影和相似性计算。它帮助模型学习到更鲁棒的状态表示，从而提高对比学习的效果。这种设计在 CURL 等对比学习方法中非常常见。

    Similar code found with 2 license types
    '''
    self.W = nn.Parameter(torch.rand(128, 128)) #

  def forward(self, x, log=False):
    '''
    return q: 动作分布
    return h:  对比学习的特征表示
    '''
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    h = torch.matmul(x, self.W_h) + self.b_h # Contrastive head
    h = nn.LayerNorm(h.shape[1])(h)
    h = F.relu(h)
    h = torch.matmul(h, self.W_c) + self.b_c # Contrastive head
    h = nn.LayerNorm(128)(h)
    # 查看md文档，因为这里说明的是实现的是Dueling DQN
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    # 这里v + a实现的是Dueling DQN的公式，也就是状态的价值加上动作的优势，组合成最终的Q值
    # 而- a.mean(1, keepdim=True) 这个懂工作是为了去除动作的平均值，避免过于依赖某个动作
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # 这里应该只是仅仅计算动作分布使用哪种softmax
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q, h

  def reset_noise(self):
    '''
    重置噪声，实现重新的进行探索
    todo 用在什么流程上
    '''
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

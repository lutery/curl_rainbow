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
from collections import namedtuple
import numpy as np
import torch


# param timestep: 当前时间步
# param state: 当前状态
# param action: 当前动作
# param reward: 当前奖励
# param nonterminal: 当前状态是否为非终止状态
# 这里的命名元组是用来存储每个时间步的状态、动作、奖励和非终止状态
Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


# Segment tree data structure where parent node values are sum/max of children node values
'''
这行注释的作用是解释 **Segment Tree** 数据结构的功能：

- **Segment Tree** 是一种树形数据结构，其中每个父节点的值是其子节点值的 **和** 或 **最大值**。
- 在这个代码中，Segment Tree 被用来高效地存储和查询优先级经验回放中的样本优先级（如 TD-Error）。
- **用途**：
  - **快速更新**：当某个样本的优先级发生变化时，可以高效地更新树结构。
  - **快速查询**：可以在对数时间复杂度内找到满足条件的样本（如按优先级比例采样）。

这使得它非常适合用于实现 **优先经验回放（Prioritized Experience Replay, PER）**。
'''
class SegmentTree():
  def __init__(self, size):
    '''
    param size: 容量
    '''
    self.index = 0 # todo 作用
    self.size = size
    self.full = False  # Used to track actual capacity
    # 构建了一个尺寸是2倍容量的树，这里存储的是每个数据的权重，父节点是子节点的权重和
    # 一层一层递归，最终根节点存储的所有数据的权重和
    # 用二叉树来存储
    self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
    # 这个存储实际的数据
    self.data = np.array([None] * size)  # Wrap-around cyclic buffer
    # todo 这个是什么？
    self.max = 1  # Initial max value to return (1 = 1^ω)

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    '''
    param index: 索引
    param value: 权重值
    '''
    # 找到当前index的父节点
    parent = (index - 1) // 2
    # 找到parent的左子节点和右子节点
    left, right = 2 * parent + 1, 2 * parent + 2
    # 左子节点的值 + 右子节点的值 = 父节点的权重值
    # todo 为啥父节点的权重值更大?
    # 因为父节点存储的是子节点的权重的总和
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    '''
    更新树的权重值
    :param index: 索引
    :param value: 权重值
    :return:
    '''
    self.sum_tree[index] = value  # Set new value
    self._propagate(index, value)  # Propagate value
    # 更新权重最大值
    self.max = max(value, self.max)

  def append(self, data, value):
    '''
    将数据添加到SegmentTree中，也就是优先级树
    :param data: 数据
    :param value: 优先级
    '''
    # 首先将数据添加到整个SegmentTree中的最后一个位置
    self.data[self.index] = data  # Store data in underlying data structure
    # 更新当前index对应的权重值
    # 使用self.index + self.size - 1这样久保证了数据对应的存储存储到树的叶子节点
    self.update(self.index + self.size - 1, value)  # Update tree
    # 可以看每次append后，index都会加1，那么self.index时记录当前缓冲区的最后一个数据的索引
    self.index = (self.index + 1) % self.size  # Update index
    # 这里的self.full是用来判断是否已经达到缓冲区的最大容量
    # 也就是当index=0时，表示缓冲区已经满了
    # 这里的self.full是用来判断是否已经达到缓冲区的最大容量
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    '''
    从索引开始向下遍历权重值
    param index: 索引
    param value: 权重值 todo
    看代码这里要找都是比当前节点的权重值小的节点
    '''
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      # 如果找到了子节点则返回当前的索尼位置
      return index
    elif value <= self.sum_tree[left]:
      # 如果当前节点大于权重，则遍历左节点
      return self._retrieve(left, value)
    else:
      # 小于权重则遍历右节点，并且右节点要减去当前节点的权重，这里一定不会负数
      # 因为大于的已经到左节点去了
      # todo 但是为啥要这么做呢？
      return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    '''
    param value: 找到符合范围value的权重值
    '''
    # 0表示从根步搜索
    index = self._retrieve(0, value)  # Search for index of item from root
    # 将权重索引转换为数据索引
    data_index = index - self.size + 1
    # 获取权重、数据索引、权重索引
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    '''
    获取指定索引位置的数据
    '''
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]

class ReplayMemory():
  def __init__(self, args, capacity):
    self.device = args.device
    self.capacity = capacity # 缓冲区容量
    self.history = args.history_length # 帧堆叠 todo
    self.discount = args.discount # 好像是奖励折扣值
    self.n = args.multi_step # 多步DQN的折扣值
    # 优先级重放缓冲区
    self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
    self.priority_exponent = args.priority_exponent
    self.t = 0  # Internal episode timestep counter
    # 用这个存储优先级貌似 todo
    self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal):
    '''
    将当前的状态、动作、奖励和终止状态添加到经验回放缓冲区中
    :param state: 当前状态
    :param action: 当前动作
    :param reward: 当前奖励
    :param terminal: 当前状态是否为终止状态
    :return:
    '''
    # 这里是将状态转换为uint8类型的张量同时将值缩放到0-255之间
    state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
    # 这里的时间步标记的是本轮游戏的第几步
    self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
    # 重置时间步
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

  # Returns a transition with blank states where appropriate
  def _get_transition(self, idx):
    '''
    获取对应索引的数据
    param idx: 索引
    return：返回一个具备历史history+未来n帧的连续缓冲区
    '''
    # 创建一个shape ： （帧堆叠历史帧+n步dqn大小尺寸的缓冲区)
    transition = np.array([None] * (self.history + self.n))
    # 提取idx所在的索引
    transition[self.history - 1] = self.transitions.get(idx)
    # 提取历史帧，如果越界了则设置为blank帧
    for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
      if transition[t + 1].timestep == 0:
        transition[t] = blank_trans  # If future frame has timestep 0
      else:
        transition[t] = self.transitions.get(idx - self.history + 1 + t)
      # 提取未来n帧，如果遇到了结束帧则设置为空帧
    for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
      if transition[t - 1].nonterminal:
        transition[t] = self.transitions.get(idx - self.history + 1 + t)
      else:
        transition[t] = blank_trans  # If prev (next) frame is terminal
    return transition

  # Returns a valid sample from a segment
  def _get_sample_from_segment(self, segment, i):
    '''
    param segment: 每个样本分配的权重基数
    param i: 第i个样本
    return: 当前选择实际权重值、数据索引、权重树索引、历史帧堆叠、未来帧堆叠、符合权重的帧的状态、符合权重帧的动作、n步dqn的折扣奖励值、未来n帧的帧堆叠、最后一帧是否结束未结束
    '''
    valid = False
    while not valid:
      # 随机选择第i个样本的权重值
      sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
      # 获取权重、数据索引、权重索引
      prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
      # Resample if transition straddled current index or probablity 0
      # 获取的数据索引要大于n步dqn的n值，因为要计算未来n步的q值
      # 并且要大于帧堆叠的帧数，因为这样才能了解之前的动作趋势吧，可能是可选的
      # 并且权重要大于0
      if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
        valid = True  # Note that conditions are valid but extra conservative around buffer index 0

    # Retrieve all required transition data (from t - h to t + n)
    # 获取指定范围的连续样本数据
    transition = self._get_transition(idx)
    # Create un-discretised state and nth next state
    # 堆叠历史帧并且归一化 todo 好像环境里面已经有帧堆叠了，这里还要继续堆叠？
    state = torch.stack([trans.state for trans in transition[:self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
    # 未来帧帧堆叠并归一化
    next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
    # Discrete action to be used as index
    # 获取当前采集帧执行动作
    action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    # n步dqn的计算，折扣值
    R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
    # Mask for non-terminal nth next states
    # 获取最后一个是否未结束的标识
    nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)

    return prob, idx, tree_idx, state, action, R, next_state, nonterminal

  def sample(self, batch_size):
    '''
    对于优先经验重放的采样

    reutrn 权重树索引、历史帧堆叠、执行的动作、这里是折扣Q值 todo和我的代码貌似不同吧、未来帧堆叠、最后一帧是否未结束、计算重要性采样权重（todo 作用）
    '''
    # 获取当前缓冲区的权重总和
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    # 将权重总优先级划分成 batch_size 个段，每个段大小相等
    segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    # 采集batch_size个数据
    batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
    # 解出数据
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
    # 堆叠所有采集数据的历史帧堆叠和未来帧堆叠
    states, next_states, = torch.stack(states), torch.stack(next_states)
    # 组合每个采集数据执行动作、这里是折扣Q值 todo和我的代码貌似不同吧、最后一帧是否未结束
    actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
    # 归一化权重优先级 将优先级标准化为 [0,1] 概率分布
    probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
    # 缓冲区如果满了，则直接返回最大的容量否则则返回实际的大小
    capacity = self.capacity if self.transitions.full else self.transitions.index
    # 计算重要性采样 todo这里是在计算什么？
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    # 归一化防止数值过大
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights


  def update_priorities(self, idxs, priorities):
    '''
    需要更新权重的权重树索引，损失loss
    '''
    # 将损失作为更新的优先级权重更新到对应idsx对应的权重树中
    # todo 了解实际运行时是怎么运作的
    priorities = np.power(priorities, self.priority_exponent)
    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    # Create stack of states
    state_stack = [None] * self.history
    state_stack[-1] = self.transitions.data[self.current_idx].state
    prev_timestep = self.transitions.data[self.current_idx].timestep
    for t in reversed(range(self.history - 1)):
      if prev_timestep == 0:
        state_stack[t] = blank_trans.state  # If future frame has timestep 0
      else:
        state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
        prev_timestep -= 1
    state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
    self.current_idx += 1
    return state

  next = __next__  # Alias __next__ for Python 2 compatibility

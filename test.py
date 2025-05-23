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
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch

from env import Env


# Test DQN
def test(args, T, dqn, val_mem, metrics, results_dir, evaluate=False):
  '''
  param args: 训练参数
  param T: 总验证步数
  param dqn: 神经网络
  param val_mem： 验证缓冲区，里面已经有了部分数据
  param metrics: 采集数据
  param results_dir； todo
  param evaluate: 是否是验证模式
  
  '''
  env = Env(args)
  env.eval() # 这里仅仅只是将环境的训练标识改为False
  metrics['steps'].append(T)
  # T_rewards记录每个episode的总奖励
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      # 选择一个动作，这里还是使用了ε-greedy策略，也就是有一定的概率是随机选择动作
      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  # Test Q-values over validation memory
  # 计算一开始的val_mem的每个状态下的Q值
  # 感觉这里是为了保持训练和验证时的接口一致才传入了一个随机的采集数据
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))

  # 平均奖励和平均Q值
  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  # 验证模式不进行保存模型
  # 如果是训练模式，则保存最好的平奖励的模型以及对应的奖励趋势还有绘制图形
  if not evaluate:
    # Save model parameters if improved
    if avg_reward > metrics['best_avg_reward']:
      metrics['best_avg_reward'] = avg_reward
      dqn.save(results_dir)

    # Append to results and save metrics
    metrics['rewards'].append(T_rewards)
    metrics['Qs'].append(T_Qs)
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Plot
    _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
    _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

  # Return average reward and Q-value
  return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)

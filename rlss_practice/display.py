#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21, 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
plt.rcParams["animation.html"] = "jshtml"

import seaborn as sns
colors = sns.color_palette('colorblind')


def display_position(image, position=None, positions=None, marker='o', marker_size=200, marker_color='b', interval=200):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)
    if positions is not None:
        y, x = positions[0]
        image = ax.scatter(x, y, marker=marker, s=marker_size, c=marker_color)

        def update(i):
            y_, x_ = positions[i]
            image.set_offsets(np.vstack((x_, y_)).T)

        return anim.FuncAnimation(fig, update, frames=len(positions), interval=interval, repeat=False)
    elif position is not None:
        y, x = position
        ax.scatter(x, y, marker=marker, s=marker_size, c=marker_color)


def display_board(image, board=None, boards=None, marker1='x', marker2='o', marker_size=200,
                  color1='b', color2='r', interval=200):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)
    if boards is not None:
        board = boards[0]
        y, x = np.where(board > 0)
        player1 = ax.scatter(x, y, marker=marker1, s=marker_size, c=color1)
        y, x = np.where(board < 0)
        player2 = ax.scatter(x, y, marker=marker2, s=marker_size, c=color2)

        def update(i):
            board_ = boards[i]
            y_, x_ = np.where(board_ > 0)
            player1.set_offsets(np.vstack((x_, y_)).T)
            y_, x_ = np.where(board_ < 0)
            player2.set_offsets(np.vstack((x_, y_)).T)

        return anim.FuncAnimation(fig, update, frames=len(boards), interval=interval, repeat=False)
    elif board is not None:
        y, x = np.where(board > 0)
        ax.scatter(x, y, marker=marker1, s=marker_size, c=color1)
        y, x = np.where(board < 0)
        ax.scatter(x, y, marker=marker2, s=marker_size, c=color2)


def plot_regret(regrets, logscale=False, lb=None,q=10):
    """
    regrets must be a dict {'agent_id':regret_table}
    """
    
    reg_plot = plt.figure()
    #compute useful stats
#     regret_stats = {}
    for i, agent_id in enumerate(regrets.keys()):
        data = regrets[agent_id]
        N, T = data.shape
        cumdata = np.cumsum(data, axis=1) # cumulative regret
        
        mean_reg = np.mean(cumdata, axis=0)
        q_reg = np.percentile(cumdata, q, axis=0)
        Q_reg = np.percentile(cumdata, 100-q, axis=0)
        
#         regret_stats[agent_id] = np.array(mean_reg, q_reg, Q_reg)
        
        plt.plot(np.arange(T), mean_reg, color=colors[i], label=agent_id)
        plt.fill_between(np.arange(T), q_reg, Q_reg, color=colors[i], alpha=0.2)
        
    if logscale:
        plt.xscale('log')
        plt.xlim(left=100)

    if lb is not None: 
        plt.plot(np.arange(T), lb, color='black', marker='*', markevery=int(T/10))
        
    plt.xlabel('time steps')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    reg_plot.show()
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.pardir)

from bandit import Bandit, Agent


class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    """行動選択を行うエージェント"""

    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        """行動の評価値を更新する
        Args:
            action (int): 選択された行動
            reward (int): 報酬
        """
        # 各マシンの評価値を更新
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        """ε-greedy 法で行動を選択する"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    bandit = NonStatBandit()
    agent = AlphaAgent(epsilon, 0.8)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()  # 行動を選択
        reward = bandit.play(action)  # スロットマシンをプレイ
        agent.update(action, reward)  # 行動の評価値を更新
        total_reward += reward

        # 結果を記録
        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    # 結果をプロット
    print("Total reward:", total_reward)

    plt.ylabel("Total rewards")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel("Average rewards")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.show()

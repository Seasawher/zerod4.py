import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.pardir)

from bandit import Bandit, Agent


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()  # 行動を選択
            reward = bandit.play(action)  # スロットマシンをプレイ
            agent.update(action, reward)  # 行動の評価値を更新
            total_reward += reward

            # 結果を記録
            rates.append(total_reward / (step + 1))
        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)

    # 結果をプロット
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_rates)
    plt.show()

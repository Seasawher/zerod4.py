import numpy as np


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """スロットマシンをプレイして報酬を返す
        Args:
            arm (int): プレイするスロットマシンの番号
        Returns:
            int: 報酬 (0 or 1)
        """
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    """行動選択を行うエージェント"""
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """行動の評価値を更新する
        Args:
            action (int): 選択された行動
            reward (int): 報酬
        """
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """ε-greedy 法で行動を選択する"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


if __name__ == "__main__":
    bandit = Bandit()
    Q = 0
    for n in range(1, 11):
        reward = bandit.play(0)  # 0 番目のスロットマシンをプレイ
        Q += (reward - Q) / n
        print(Q)

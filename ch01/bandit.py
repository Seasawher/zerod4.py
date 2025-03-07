import numpy as np


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


if __name__ == "__main__":
    bandit = Bandit()
    for i in range(10):
        print(bandit.play(0))

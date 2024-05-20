from collections import deque

import numpy as np

from .utils import MinTree, SumTree, Transition, discount_cumsum


class Buffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size

    def append(self, transition: Transition):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def __repr__(self):
        main_str = f"{self.__class__.__name__}(memory_size={self.memory_size})"
        return main_str


class ReplayBuffer(Buffer):
    """
    観測・行動は-1 ~ 1に正規化されているものとして扱う
    """

    def __init__(self, memory_size):
        super().__init__(memory_size)
        self.memory: deque[Transition] = deque([], maxlen=memory_size)

    def append(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states = np.array([self.memory[index].state for index in batch_indexes])
        next_states = np.array([self.memory[index].next_state for index in batch_indexes])
        rewards = np.array([self.memory[index].reward for index in batch_indexes])
        actions = np.array([self.memory[index].action for index in batch_indexes])
        dones = np.array([self.memory[index].done for index in batch_indexes])
        return {"states": states, "next_states": next_states, "rewards": rewards, "actions": actions, "dones": dones}


class PrioritizedReplayBuffer(Buffer):
    def __init__(self, memory_size, alpha=0.6, beta=0.4, total_steps=2500000):
        super().__init__(memory_size)
        self.memory: deque[Transition] = deque([], maxlen=memory_size)
        self.priorities = SumTree(memory_size)
        self.min_priority = MinTree(memory_size)
        self.alpha = alpha
        self.beta_scheduler = lambda steps: beta + (1 - beta) * steps / total_steps
        self.epsilon = 0.01
        self.max_priority = 1.0
        self.ptr = 0
        self.indices = []

    def __len__(self):
        return len(self.memory)

    def append(self, transition: Transition):
        if self.ptr == self.memory_size:
            self.ptr = 0
        try:
            self.memory[self.ptr] = transition
        except IndexError:
            self.memory.append(transition)
        self.priorities[self.ptr] = self.max_priority
        self.min_priority[self.ptr] = self.max_priority
        self.ptr += 1

    def sample(self, batch_size, steps):
        indices = np.array([self.priorities.sample() for _ in range(batch_size)])
        self.indices = indices

        n = len(self.memory)
        beta = self.beta_scheduler(steps)
        prob_min = self.min_priority.min() / self.priorities.sum()
        max_weight = (prob_min * n) ** (-beta)
        weights = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            idx = indices[i]
            prob = self.priorities[idx] / self.priorities.sum()
            weight = (prob * n) ** (-beta)
            weights[i] = weight / max_weight

        states = np.array([self.memory[index].state for index in indices])
        next_states = np.array([self.memory[index].next_state for index in indices])
        rewards = np.array([self.memory[index].reward for index in indices])
        actions = np.array([self.memory[index].action for index in indices])
        dones = np.array([self.memory[index].done for index in indices])

        return {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "actions": actions,
            "dones": dones,
            "weights": weights,
        }

    def update_priority(self, td_erros):
        priorities: np.ndarray = (np.abs(td_erros) + self.epsilon) ** self.alpha
        for idx, priority in zip(self.indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, priorities.max())


class PPOBuffer(Buffer):
    def __init__(self, memory_size, obs_dim, act_dim, gamma=0.99, lam=0.95):
        super().__init__(memory_size)
        self.obs = np.zeros((memory_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((memory_size, act_dim), dtype=np.float32)
        self.adv = np.zeros(memory_size, dtype=np.float32)
        self.rew = np.zeros(memory_size, dtype=np.float32)
        self.ret = np.zeros(memory_size, dtype=np.float32)
        self.val = np.zeros(memory_size, dtype=np.float32)
        self.logp = np.zeros(memory_size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx = 0, 0

    def append(self, obs, act, rew, val, logp):
        assert self.ptr < self.memory_size
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew[path_slice], last_val)
        vals = np.append(self.val[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[:-1] - vals[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def sample(self):
        assert self.ptr == self.memory_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean = np.mean(self.adv)
        adv_diff = self.adv - adv_mean
        adv_std = np.mean(adv_diff**2)
        self.adv = adv_diff / adv_std
        return {"obs": self.obs, "act": self.act, "ret": self.ret, "adv": self.adv, "logp": self.logp}

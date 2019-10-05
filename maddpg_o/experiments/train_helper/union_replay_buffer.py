import numpy as np
import random
import os
import joblib

class UnionReplayBuffer(object):
    def __init__(self, size, n_items, n_agents):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = [[None for _ in range(n_agents)] for _ in range(n_items)]
        self._maxsize = int(size)
        self._next_idx = 0
        self.n_items = n_items
        self.n_agents = n_agents
        self.len = 0
        self.first = True

    def __len__(self):
        return self.len

    def save(self, save_dir):
        save_path = os.path.join(save_dir, "union_buffer.data")
        joblib.dump((self._storage, self._next_idx, self.len, self.first), save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, "union_buffer.data")
        self._storage, self._next_idx, self.len, self.first = joblib.load(load_path)

    def clear(self):
        self._storage = [[None for _ in range(self.n_agents)] for _ in range(self.n_items)]
        self._next_idx = 0
        self.first = True
        self.len = 0

    def add(self, data):
        # data : (m, n, shape)

        if self._next_idx >= self.len:
            if self.first:
                for i in range(self.n_items):
                    for j in range(self.n_agents):
                        s = data[i][j].shape
                        self._storage[i][j] = np.zeros(shape=(self._maxsize, *s))
                self.first = False
            # else:
                # for i in range(self.m):
                #     for j in range(self.n):
                #         self._storage[i][j] = self._storage[i][j] + [data[i][j]]
            self.len += 1
        # else:
        for i in range(self.n_items):
            for j in range(self.n_agents):
                self._storage[i][j][self._next_idx] = data[i][j]
        self._next_idx = (self._next_idx + 1) % self._maxsize
    #
    # def _encode_sample(self, idxes):
    #     obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
    #     for i in idxes:
    #         data = self._storage[i]
    #         obs_t, action, reward, obs_tp1, done = data
    #         obses_t.append(np.array(obs_t, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         obses_tp1.append(np.array(obs_tp1, copy=False))
    #         dones.append(done)
    #     return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
    #
    # def encode_sample_simple(self, idxes, n):
    #     obses_t, actions, rewards, obses_tp1, dones = [[] for _ in range(n)], \
    #                                                   [[] for _ in range(n)], \
    #                                                   [[] for _ in range(n)], \
    #                                                   [[] for _ in range(n)], \
    #                                                   [[] for _ in range(n)]
    #     for i in idxes:
    #         data = self._storage[i]
    #         obs_t, action, reward, obs_tp1, done = data
    #         for j in range(n):
    #             obses_t[j].append(obs_t[j])
    #             actions[j].append(action[j])
    #             obses_tp1[j].append(obs_tp1[j])
    #             rewards[j].append(reward[j])
    #             dones[j].append(done[j])
    #
    #     for j in range(n):
    #         obses_t[j] = np.array(obses_t[j])
    #         actions[j] = np.array(actions[j])
    #         obses_tp1[j] = np.array(obses_tp1[j])
    #         rewards[j] = np.array(rewards[j])
    #         dones[j] = np.array(dones[j])
    #     return obses_t, actions, rewards, obses_tp1, dones

    def make_index(self, batch_size):
        return np.random.randint(self.len, size=batch_size)

    def sample_index(self, idxes):
        # ret = [[[] for _ in range(self.n)] for _ in range(self.m)]
        # for i in range(self.m):
        #     for j in range(self.n):
        #         for k in idxes:
        #             ret[i][j].append(self._storage[i][j][k])
        #         ret[i][j] = np.array(ret[i][j])
        return [[self._storage[i][j][idxes] for j in range(self.n_agents)] for i in range(self.n_items)]

    # def make_latest_index(self, batch_size):
    #     idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
    #     np.random.shuffle(idx)
    #     return idx
    #
    # def sample_index(self, idxes):
    #     return self._encode_sample_simple(idxes)
    #
    # def sample(self, batch_size):
    #     """Sample a batch of experiences.
    #
    #     Parameters
    #     ----------
    #     batch_size: int
    #         How many transitions to sample.
    #
    #     Returns
    #     -------
    #     obs_batch: np.array
    #         batch of observations
    #     act_batch: np.array
    #         batch of actions executed given obs_batch
    #     rew_batch: np.array
    #         rewards received as results of executing act_batch
    #     next_obs_batch: np.array
    #         next set of observations seen after executing act_batch
    #     done_mask: np.array
    #         done_mask[i] = 1 if executing act_batch[i] resulted in
    #         the end of an episode and 0 otherwise.
    #     """
    #     if batch_size > 0:
    #         idxes = self.make_index(batch_size)
    #     else:
    #         idxes = range(0, len(self._storage))
    #     return self._encode_sample(idxes)
    #
    # def collect(self):
    #     return self.sample(-1)

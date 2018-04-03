import numpy as np
import random


class Memory:
    def __init__(self, size, initial_value=None):
        self.data = [initial_value] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, memory_item):
        self.data[self.end] = memory_item
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_state(self, index=None, n=4):
        if index is None:
            index = self.end

        if len(self) >= n:
            start = index - n

            if start >= 0:
                # print("Frames used: {} - {}".format(start, index))
                observations = [memory_item.observation for memory_item in self.data[start:index]]
            else:
                start = start % len(self.data)
                # print("Frames used: {} - {}".format(start, index))
                observations = [memory_item.observation for memory_item in self.data[:index] + self.data[start:]]

            return np.stack(observations, axis=2)

        return None

    def get_batch(self, k=32):
        ind = random.sample(range(4, len(self)), k)
        states = [self.get_state(index=i) for i in ind]
        next_states = [self.get_state(index=i + 1) for i in ind]

        actions, rewards, dones, infos = zip(*[(self.data[i].action,
                                                self.data[i].reward,
                                                self.data[i].done,
                                                self.data[i].info)
                                               for i in ind])

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        rewards = np.stack(rewards, axis=0)
        dones = np.stack(dones, axis=0)

        return states, actions, next_states, rewards, dones, infos

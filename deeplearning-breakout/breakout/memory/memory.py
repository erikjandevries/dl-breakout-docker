import numpy as np
import random

from .ringbuffer import RingBuffer


class Memory(RingBuffer):
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

    def get_batch(self, k=32, include_last=False):
        if include_last:
            ind = [self.end - 1] + random.sample(range(4, len(self)), k - 1)
        else:
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

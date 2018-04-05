import pickle


class RingBuffer():
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

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=4)

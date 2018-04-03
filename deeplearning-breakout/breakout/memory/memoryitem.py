class MemoryItem:
    def __init__(self, observation, action, next_observation, reward, done, info):
        self.observation = observation
        self.action = action
        self.next_observation = next_observation
        self.reward = reward
        self.done = done
        self.info = info

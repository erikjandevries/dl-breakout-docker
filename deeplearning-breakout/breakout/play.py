import argparse
import gym
from time import sleep

from agent import Agent
from memory import MemoryItem
from preprocessing import preprocess

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler(str('./data/playing.log'))
fh.setFormatter(formatter)
log.addHandler(ch)
log.addHandler(fh)


def main(load_model=None, max_episodes=5):
    # env = gym.make('BreakoutDeterministic-v4')
    env = gym.make('Breakout-v4')
    agent = Agent(
        env=env,
        memory_size=100
    )

    if load_model:
        log.info("Loading model file: {}".format(load_model))
        agent.load_model(load_model)

    for n_episodes in range(max_episodes):
        next_observation = preprocess(env.reset())
        env.render()

        episode_done = False
        total_reward = 0
        # agent._ba = 0

        while not episode_done:
            # sleep for the duration of the frame so we can see what happens
            sleep(1. / 30)

            observation = next_observation

            if (load_model is not None) & (len(agent.memory) >= 4):
                action = agent.pick_best_action()
            else:
                action = agent.pick_random_action()

            next_observation, reward, episode_done, info = env.step(action)
            next_observation = preprocess(next_observation)
            agent.memory.append(MemoryItem(observation, action, next_observation, reward, episode_done, info))

            env.render()

            total_reward += reward
            # print("E {} - r {} - m {} - BAs {}".format(n_episodes, total_reward, len(agent.memory), agent._ba), end="\r")

        log.info("E {} - r {}".format(n_episodes, total_reward))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model",
                        dest="load_model",
                        help="Pre-trained model to load")
    args = parser.parse_args()

    main(load_model=args.load_model)

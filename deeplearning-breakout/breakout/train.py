import argparse

import gym
# import numpy as np
import random

from agent import Agent
from memory import MemoryItem
from preprocessing import preprocess

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler(str('./data/model_training.log'))
fh.setFormatter(formatter)
log.addHandler(ch)
log.addHandler(fh)


def get_epsilon_for_iteration(iteration):
    if iteration > 1e6:
        return 0.1
    return 1 - (0.9 * iteration / 1e6)


def main(max_episodes, max_frames, memory_size, load_model, save_model):
    env = gym.make('BreakoutDeterministic-v4')
    agent = Agent(
        env=env,
        memory_size=memory_size
    )

    if load_model:
        log.info("Loading model file: {}".format(load_model))
        agent.load_model(load_model)

    agent.model.summary()

    try:
        n_episodes = 0
        n_frames = 0

        log.info("Training for a maximum of {} episodes and {} frames".format(max_episodes, max_frames))

        while (n_episodes < max_episodes and
               n_frames < max_frames):
            n_episodes += 1

            next_observation = preprocess(env.reset())

            episode_done = False
            total_reward = 0
            agent._ba = 0

            while not episode_done:
                n_frames += 1
                observation = next_observation

                if random.uniform(0, 1) <= get_epsilon_for_iteration(n_frames):
                    action = agent.pick_random_action()
                else:
                    action = agent.pick_best_action()

                next_observation, reward, episode_done, info = env.step(action)
                next_observation = preprocess(next_observation)

                agent.memory.append(MemoryItem(observation, action, next_observation, reward, episode_done, info))

                if len(agent.memory) > 64:
                    agent.train_step()

                total_reward += reward

            log.info("E {} - f {} - r {} - BAs {}".format(n_episodes, n_frames, total_reward, agent._ba))
            if n_episodes % 100 == 0:
                env.render()
                log.info("Saving model file: {}".format(save_model))
                agent.save_model(save_model)

    except KeyboardInterrupt:
        print("\nUser interrupted training...")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes",
                        dest="max_episodes",
                        type=int,
                        default=5,
                        help="The maximum number of episodes to train for")
    parser.add_argument("--max-frames",
                        dest="max_frames",
                        type=int,
                        default=50000000,
                        help="The maximum number of frames to train for")
    parser.add_argument("--memory-size",
                        dest="memory_size",
                        type=int,
                        default=1000000,
                        help="The number of frames to keep in memory (default: 1,000,000)")
    parser.add_argument("--load-model",
                        dest="load_model",
                        help="Pre-trained model to load")
    parser.add_argument("--save-model",
                        dest="save_model",
                        default="./data/model.h5",
                        help="Save model to file")
    args = parser.parse_args()

    main(
        max_episodes=args.max_episodes,
        max_frames=args.max_frames,
        memory_size=args.memory_size,
        load_model=args.load_model,
        save_model=args.save_model
    )

import argparse

import gym
import numpy as np
import random
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
fh = logging.FileHandler(str('./data/model_training.log'))
fh.setFormatter(formatter)
log.addHandler(ch)
log.addHandler(fh)


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def get_epsilon_for_iteration(n_frames, n_frames_in_episode, avg_frames_per_episode):
    # Base probability of experimentation
    if n_frames < 1e3:
        p = 1
    elif n_frames < 1e6:
        # # Up to 1M frames played, gradually decay from 100% to 10%
        # p = 1 - (0.9 * n_frames / 1e6)
        # Up to 1M frames played, gradually decay from 50% to 10%
        p = 0.5 - (0.4 * n_frames / 1e6)
    else:
        # If at least 1M frames played, experiment only 10% of the time
        p = 0.1

    if n_frames > 1e3:
        # Start an episode with less experimentation
        # Finish with more experimentation
        # Gradually increase from 0.1 * p to 1.9 * p
        p *= (0.2 + 1.8 * n_frames_in_episode / avg_frames_per_episode)
    return p


def get_mean_frames_per_episode(frames_per_episode, last_n=50):
    if len(frames_per_episode) > last_n:
        return mean(frames_per_episode[-last_n:])
    elif len(frames_per_episode) > 0:
        return mean(frames_per_episode)
    return 200


def main(max_episodes, max_frames, memory_size, load_model, save_model, load_memory, save_memory):
    # env = gym.make('BreakoutDeterministic-v4')
    env = gym.make('Breakout-v4')
    agent = Agent(
        env=env,
        memory_size=memory_size
    )

    agent.model.summary()

    try:
        n_episodes = 0
        n_frames = 0
        frames_per_episode = []

        if load_model:
            agent.load_model(load_model)

        if load_memory:
            agent.load_memory(load_memory)
            # skip the framecount ahead - to work with the appropriate epsilon
            n_frames = len(agent.memory)

        log.info("Training for a maximum of {} episodes and {} frames".format(max_episodes, max_frames))

        while (n_episodes < max_episodes and
               n_frames < max_frames):
            n_episodes += 1

            mean_frames_per_episode = get_mean_frames_per_episode(frames_per_episode)

            next_observation = preprocess(env.reset())

            episode_done = False
            total_reward = 0
            agent._ba = 0
            n_frames_in_episode = 0

            while not episode_done:
                n_frames += 1
                n_frames_in_episode += 1
                observation = next_observation

                epsilon = get_epsilon_for_iteration(n_frames, n_frames_in_episode, mean_frames_per_episode)

                if random.uniform(0, 1) <= epsilon:
                    action = agent.pick_random_action()
                else:
                    action = agent.pick_best_action()

                next_observation, reward, episode_done, info = env.step(action)
                next_observation = preprocess(next_observation)

                agent.memory.append(MemoryItem(observation, action, next_observation, reward, episode_done, info))

                if len(agent.memory) > 65:
                    agent.train_step()

                total_reward += reward

            frames_per_episode.append(n_frames_in_episode)

            log.info("E {} - f {} - r {} - BAs {} - mfpe {}".format(
                n_episodes, n_frames_in_episode, total_reward, agent._ba, mean_frames_per_episode))
            if n_episodes % 100 == 0:
                # env.render()
                agent.save_model(save_model)

    except KeyboardInterrupt:
        print("\nUser interrupted training...")
    finally:
        agent.save_model(save_model)
        agent.save_memory(save_memory)

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
    parser.add_argument("--load-memory",
                        dest="load_memory",
                        help="Pre-filled memory to load")
    parser.add_argument("--save-memory",
                        dest="save_memory",
                        default="./data/memory.pkl",
                        help="Save memory to file")
    args = parser.parse_args()

    main(
        max_episodes=args.max_episodes,
        max_frames=args.max_frames,
        memory_size=args.memory_size,
        load_model=args.load_model,
        save_model=args.save_model,
        load_memory=args.load_memory,
        save_memory=args.save_memory
    )

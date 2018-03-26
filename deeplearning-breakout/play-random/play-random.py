import gym
from time import sleep


def main():
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()

    is_done = False
    while not is_done:
        # sleep for the duration of the frame so we can see what happens
        sleep(1./30)
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        # Render
        env.render()


if __name__ == '__main__':
    main()

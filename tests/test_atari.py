import gym
import gym.wrappers as w

import cma_es
import gym_wrappers


def test_atari():
    env = gym.make('PongNoFrameskip-v4')
    env = w.atari_preprocessing.AtariPreprocessing(env.unwrapped, terminal_on_life_loss=True)
    obs = env.reset()
    env.render()
    done = False

    while not done:
        obs, r, done, info = env.step(cma_es.sample())
        env.render()


def test_wrapper():
    env = gym.make('Pong-v0')
    env = gym_wrappers.RewardCountLimit(env, max_reward_count=5)
    obs = env.reset()
    env.render()
    done = False

    while not done:
        obs, r, done, info = env.step(cma_es.sample())
        env.render()

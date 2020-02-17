import gym


class RewardCountLimit(gym.Wrapper):
    def __init__(self, env, max_reward_count=None):
        """
        Returns done once a number of nonzero rewards have been received
        :param env: the env to wrap
        :param max_reward_count: the
        """
        super().__init__(env)
        self.max_reward_count = max_reward_count
        self.reward_count = 0

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self.reward_count += 0 if reward == 0 else 1
        if self.reward_count >= self.max_reward_count:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
import gym
from gym_snake.envs.snake import Controller, Discrete
import matplotlib.pyplot as plt
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15, 15], unit_size=10, unit_gap=1, snake_size=3, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init

    def step(self, action):
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size,
                                     random_init=self.random_init)
        self.last_obs = self.controller.grid.grid
        return self.last_obs

    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = plt.imshow(self.last_obs)
        else:
            self.viewer.set_data(self.last_obs)
        plt.pause(0.1)
        plt.draw()

    def seed(self, x):
        pass


class SnakeEnvRotate(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15, 15], unit_size=1, unit_gap=0, snake_size=3, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.viewer = None
        self.action_space = Discrete(4)
        self.random_init = random_init

    def step(self, action):
        action = self._rotate_action(action)
        self.last_obs, rewards, done, info = self.controller.step(action)
        return self._get_obs(), rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size,
                                     random_init=self.random_init)
        return self._get_obs()

    def render(self, mode='human', close=False):
        if self.viewer is None:
            self.viewer = plt.imshow(self.last_obs)
        else:
            self.viewer.set_data(self.last_obs)
        plt.pause(0.1)
        plt.draw()

    def _rotate_action(self, action):
        try:
            return (action - self.controller.snake.direction) % 4
        except AttributeError:
            return 0

    def _get_obs(self):
        grid = self._get_rotated_grid()
        return self._convert_grid_to_bool(grid)

    def _get_rotated_grid(self):
        try:
            d = self.controller.snake.direction
            g = self.controller.grid.grid
            return np.rot90(g, d, axes=(0, 1))
        # Catching error when snek is dead and therefore is none. (Returns black screen)
        except AttributeError:
            return np.zeros([self.grid_size[0], self.grid_size[1], 3])

    def _convert_grid_to_bool(self, grid):
        s = grid == [255, 255, 255]
        # Find the body positions
        b = (grid == [1, 0, 0])[:, :, 0]
        return np.dstack((s, b))

    def seed(self, x):
        pass
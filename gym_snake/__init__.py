from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
)

register(
    id='snake-rotate-visual-v0',
    entry_point='gym_snake.envs:SnakeEnvRotateVisual',
)

register(
    id='snake-rotate-v0',
    entry_point='gym_snake.envs:SnakeEnvRotate',
)
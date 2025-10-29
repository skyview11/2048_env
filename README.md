# 2048_env
Environment of 2048 game for machine learning



## How to use

1. call object
``` python

from env import Env2048

env = Env2048(n_parallel=10, board_size=(4, 4))
```
`n_parallel`: the number of game run parallely
`board_size`: board size of 2048 game

2. run step

```python
info = env.step(actions)
```
this function takes a single step for every parallel single games

`actions`: the list of actions each action is encoded to 0, 1, 2 and 3 and each means up, left, down and right move The size of the list should be same with `n_parallel` defined above

3. reset
``` python
env.reset()
```

You can reset your environment to initial setting

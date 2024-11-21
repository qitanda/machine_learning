# RL in Gym

## 环境

- pytorch
- gymnasium

> see [here](https://github.com/Farama-Foundation/Gymnasium) for detail.

1. run `pip install gymnasium` to install the base Gymnasium library.
2. run `sudo apt install swig` and  `pip install "gymnasium[box2d]"` to install LunarLander environment.

> 所有环境信息可在[官网](https://gymnasium.farama.org/environments/box2d/lunar_lander/)查询.

> Other Envirment: run `pip install "gymnasium[atari]"` and `pip install "gymnasium[accept-rom-license]"` to install Atari environment. (see [here](https://gymnasium.farama.org/environments/atari/) for detail)

## Demo

`demo.py` 为示例代码，主要看 `train` 和 `eval` 函数。

- Algorithm: DQN
- Env: LunarLander-v2

## 要求

通过修改demo中的算法，参数，奖励设置使得飞船平稳落地。

- 运行 `eval` 函数能平稳落地, 不限制使用各种算法.
- 提交算法代码
- 提交训练生成的文件(文件夹`LunarLnader`里的东西)

> 所有模型，奖励文件提交一份就行，一个小组交一份！！！！

> 命名：组号-第三次作业

> 提交到邮箱: 22S053079@stu.hit.edu.cn


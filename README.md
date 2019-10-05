
# Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning

This is the code for implementing the MADDPG algorithm presented in the paper:
[Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=SJxbHkrKDH).
It is configured to be run in conjunction with environments from the(https://github.com/qian18long/epciclr2020/tree/master/mpe_local).
Note: this codebase has been restructured since the original paper, and the results may
vary from those reported in the paper.

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

## Case study: Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the(https://github.com/qian18long/epciclr2020/tree/master/mpe_local). It is based on(https://github.com/openai/multiagent-particle-envs)



## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"grassland"`)


- `--map-size`: The size of the environment. 1 if normal and 2 otherwise. (default: `"normal"`)

- `sight`: The agent's visibility radius. (default: `100`)

- `no-wheel`: ???

- `alpha`: Reward shared weight. (default: `0.0`)

- `show-attention`: ???

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `200000`)

- `--num-good`: number of good agents in the scenario (default: `2`)

- `--num-adversaries`: number of adversaries in the environment (default: `2`)

- `--num-food`: number of food(resources) in the scenario (default: `4`)

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"maddpg"`; options: {`"att-maddpg"`, `"maddpg"`, `"PC"`, `"mean-field"`})

- `--adv-policy`: algorithm used for the adversary policies in the environment
(default: `"maddpg"`; options: {`"att-maddpg"`, `"maddpg"`, `"PC"`, `"mean-field"`})

### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--good-num-units`: number of units in the MLP of good agents, if not providing it will be num-units.

- `--adv-num-units`: number of units in the MLP of adversarial agents, if not providing it will be num-units.

- `--n_cpu_per_agent`: cpu usage per agent (default: `1`)

- `good-share-weights`: good agents share weights of the agents encoder within the model.

- `adv-share-weights`: adversarial agents share weights of the agents encoder within the model.

### Checkpointing

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/test/"`)

- `--train-rate`: ???? (default: `100`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--checkpoint-rate`: ???? (default: `0`)

- `--load-dir`: directory where training state and model are loaded from (default: `"test"`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--eval`: ???? (default: `0`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--save-gif-data`: ???? (default: `0`)

- `--render-gif`: ???? (default: `0`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--use-gpu`: ???? (default: `0`)

- `--n-envs`: ???? (default: `0`)

- `--save-summary`: ???? (default: `0`)

- `--timeout`: ???? (default: `0`)

## Code structure

- `./experiments/train.py`: contains code for training MADDPG on the MPE

- `./maddpg/trainer/maddpg.py`: core code for the MADDPG algorithm

- `./maddpg/trainer/replay_buffer.py`: replay buffer code for MADDPG

- `./maddpg/common/distributions.py`: useful distributions used in `maddpg.py`

- `./maddpg/common/tf_util.py`: useful tensorflow functions used in `maddpg.py`



## Paper citation

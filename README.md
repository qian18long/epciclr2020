
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

- `--sight`: The agent's visibility radius. (default: `100`)

- `--no-wheel`: ???

- `--alpha`: Reward shared weight. (default: `0.0`)

- `--show-attention`: ???

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

- `--good-share-weights`: good agents share weights of the agents encoder within the model.

- `--adv-share-weights`: adversarial agents share weights of the agents encoder within the model.

- `--use-gpu`: Use GPU to for training (default: `False`)

### Checkpointing

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/test/"`)

- `--train-rate`: (default: `100`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--checkpoint-rate`: (default: `0`)

- `--load-dir`: directory where training state and model are loaded from (default: `"test"`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--save-gif-data`: Save the gif examples to the save-dir (default: `False`)

- `--render-gif`: Render the gif in the load-dir (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--n-envs`: (default: `1`)

- `--save-summary`: (default: `False`)

- `--timeout`: (default: `0.02`)

## Code structure

- `.maddpg_o/experiments/train_helper/train_helpers.py`: contains code for training MADDPG, Att-MADDPG, mean-field, Vanilla PC and EPC on the MPE

- `.maddpg_o/experiments/train_normal.py`: apply the train_helpers.py for MADDPG, Att-MADDPG and mean-field training

- `.maddpg_o/experiments/train_normal.py`: apply the population curriculum in train_helpers.py to add an agent in model of load_dir.

- `.maddpg_o/experiments/train_x2.py`: apply the population curriculum in train_helpers.py to duplicate agents in model of load_dir.

- `.maddpg_o/experiments/train_mix_match.py`: Mix match of the good agents in '--sheep-init-load-dirs' and adversarial agents in '--wolf-init-load-dirs' for model agents evaluation.

- `.maddpg_o/experiments/compete.py`: Mix match within all agent groups in '--competitor-load-dirs' to evaluate all agent groups' performances for EVOLUTIONARY SELECTION.

- `./maddpg_o/maddpg_local/micro/maddpg.py`: core code for the MADDPG based algorithm

- `./maddpg_o.experiments.train_helper.union_replay_buffer`: replay buffer code

- `./maddpg_o/maddpg_local/common/distributions.py`: useful distributions used in `maddpg.py`

- `./maddpg_o/maddpg_local/common/tf_util.py`: useful tensorflow functions used in `maddpg.py`



## Paper citation

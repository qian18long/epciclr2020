
# Multi-Agent Particle Environment

A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
Used in the paper [Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=SJxbHkrKDH). The code is changed based on(https://github.com/openai/multiagent-particle-envs)

## Code structure

- `./multiagent/environment.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.)

- `./multiagent/core.py`: contains classes for various objects (Entities, Landmarks, Agents, etc.) that are used throughout the code.

- `./multiagent/rendering.py`: used for displaying agent behaviors on the screen.

- `./multiagent/policy.py`: contains code for interactive policy based on keyboard input.

- `./multiagent/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./multiagent/scenarios/`: folder where various scenarios/ environments are stored. scenario code consists of several functions:
    1) `make_world()`: creates all of the entities that inhabit the world (landmarks, agents, etc.), assigns their capabilities (whether they can communicate, or move, or both).
     called once at the beginning of each training session
    2) `reset_world()`: resets the world by assigning properties (position, color, etc.) to all entities in the world
    called before every episode (including after make_world() before the first episode)
    3) `reward()`: defines the reward function for a given agent
    4) `observation()`: defines the observation space of a given agent
    5) (optional) `benchmark_data()`: provides diagnostic data for policies trained on the environment (e.g. evaluation metrics)

### Creating new environments

You can create new scenarios by implementing the first 4 functions above (`make_world()`, `reset_world()`, `reward()`, and `observation()`).

## List of environments


| Env name in code (name in paper) | Competitive or Cooperative | Notes |
| --- | --- | --- |
| `grassland.py` | Fully Competitive | The Grassland Game. Sheep (blue) are two times as fast as wolves (red). We also have a fixed amount of L grass pellets (food for sheep) as green landmarks. A wolf will be rewarded when it collides with (eats) a sheep, and the (eaten) sheep will obtain a negative reward and becomes inactive (dead). A sheep will be rewarded when it comes across a grass pellet and the grass will be collected and respawned in another random position. Note that in this survival game, each individual agent has its own reward and does not share rewards with others. |
| `adversarial.py` | Partial Cooperative Partial Competitive | The Adversarial Battle Game. two teams of agents (i.e., Ω = 2 for each team) competing for the resources. Both teams have the same number of agents (N1 = N2). When an agent collects a unit of resource, the resource will be respawned and all the agents in its team will receive a positive reward. Furthermore, if there are more than two agents from team 1 collide with one agent from team 2, the whole team 1 will be rewarded while the trapped agent from team 2 will be deactivated (dead) and the whole team 2 will be penalized, and vice versa. |
| `food_collect.py` | Fully Cooperative| The Food Collection Game. N fully cooperated agents and N food locations. The agents need to collaboratively occupy as many food locations as possible within the game horizon. Whenever a food is occupied by any agent, the whole team will get a reward of 6/N in that timestep for that food. The more food occupied, the more rewards the team will collect. |

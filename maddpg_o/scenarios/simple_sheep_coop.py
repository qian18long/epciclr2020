import numpy as np
from mpe_local.multiagent.core import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import os

SIGHT = 0.5

class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, sight, no_wheel, ratio):
        self.n_good = n_good
        self.n_adv = 0
        self.n_landmarks = n_landmarks
        self.n_food = n_food
        self.n_forests = n_forests
        self.alpha = alpha
        self.sight = sight
        self.no_wheel = no_wheel
        print(sight,"sheep_coop!!!!!!!")
        print(alpha,"################alpha v2##############")

    def make_world(self):

        world = World()
        # set any world properties first
        world.collaborative = True
        world.dim_c = 2
        num_good_agents = self.n_good
        num_adversaries = self.n_adv
        world.num_good_agents = num_good_agents
        world.num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = self.n_landmarks
        num_food = self.n_food
        num_forests = self.n_forests
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.06 if agent.adversary else 0.06
            agent.accel = 4.0 if agent.adversary else 4.0
            if agent.adversary:
                agent.showmore = np.zeros(num_good_agents)
            else:
                agent.showmore = np.zeros(num_food)
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 3 if agent.adversary else 3
            agent.live = 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.sight = 1
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0
            landmark.boundary = False
        # make initial conditions
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.sight = 1
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.sight = 1
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            landmark.boundary = False
        world.landmarks += world.food
        world.landmarks += world.forests
        self.reset_world(world)
        return world


    def reset_world(self, world):
        seed = int.from_bytes(os.urandom(4), byteorder='little')
        # print("reseed to", seed)
        np.random.seed(seed)

        # random properties for agents
        #########

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.live = 1
            if agent.adversary:
                agent.showmore = np.zeros(world.num_good_agents)
            else:
                agent.showmore = np.zeros(world.num_adversaries)
            # agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def done(self, agent, world):
        return 0

    def info(self, agent, world):
        time_grass = []
        time_live = []

        mark_grass = 0
        if agent.live:
            time_live.append(1)
            for food in world.food:
                if self.is_collision(agent, food):
                    mark_grass = 1
                    break
        else:
            time_live.append(0)
        if mark_grass:
            time_grass.append(1)
        else:
            time_grass.append(0)

        return np.concatenate([np.array(time_grass)]+[np.array(time_live)])

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        main_reward = self.reward_all_in_once(agent, world)
        return main_reward

    def reward_all_in_once(self, agent, world):
        num_agents = len(world.agents)
        reward_n = np.zeros(num_agents)
        alpha_sharing = self.alpha
        shape = 0

        for i, agent_new in enumerate(world.agents):
            # shape of food:
            distance_min = min([np.sqrt(np.sum(np.square(food.state.p_pos - agent_new.state.p_pos))) for food in world.food])
            if distance_min<self.sight and not self.no_wheel:
                reward_n[i] -= 0.1 * distance_min

            # shape to encourage collision:
            if shape:
                distance_min = min(np.sqrt(np.sum(np.square(agent_new.state.p_pos - good.state.p_pos)))for good in world.agents)
                if distance_min<self.sight and not self.no_wheel:
                    reward_n[i] -= 0.04*distance_min
        # eat food reward
        food_id = []
        number_eat = 0
        for i_food, food in enumerate(world.food):
            number_eat = 0
            for agent_new in world.agents:
                if self.is_collision(agent_new, food):
                    number_eat += 1
            if number_eat>=2:
                reward_buffer = 2*4/num_agents*np.ones(num_agents)
                reward_n += reward_buffer
                food_id.append(i_food)

        for i_food in food_id:
            world.food[i_food].state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
        return list(reward_n)


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []

        for entity in world.landmarks:
            distance = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
            if distance > self.sight:
                entity_pos.append([0,0,0])
            else:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                entity_pos.append([1])
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        other_live = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            distance = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            # print(distance,'distance')
            # print(other.live, 'other_live')
            if distance > self.sight or (not other.live):
                other_pos.append([0,0])
                other_vel.append([0,0])
                other_live.append([0])
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel)
                other_live.append(np.array([other.live]))
        result = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.live])] + entity_pos + other_pos + other_vel + other_live)
        # print(result.shape,"shape#################")
        return result

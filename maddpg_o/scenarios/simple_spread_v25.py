import numpy as np
from mpe_local.multiagent.core import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import os

SIGHT = 100
ALPHA = 1

def softmax_dis(x):
    x = np.asarray(x)
    dis_min = np.min(x)
    return dis_min




class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, sight, no_wheel, ratio):
        self.n_good = n_good
        self.n_landmarks = n_landmarks
        self.n_food = n_food
        self.n_forests = n_forests
        self.alpha = alpha
        self.sight = sight
        self.no_wheel = no_wheel
        print(sight,"sight___simple_spread_v25")
        print(alpha,"alpha######################")

    def make_world(self):
        world = World()
        # set any world properties first
        world.collaborative = True
        world.dim_c = 2
        num_good_agents = self.n_good
        world.num_good_agents = num_good_agents
        num_agents = num_good_agents
        num_landmarks = self.n_landmarks
        num_food = self.n_food
        num_forests = self.n_forests
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = False
            agent.size = 0.05
            agent.accel = 4.0
            agent.showmore = np.zeros(num_food)
            agent.max_speed = 4
            agent.live = 1
            agent.mindis = 0
            agent.time = 0
            agent.occupy = 0
        
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0
            landmark.boundary = False
        # make initial conditions
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
            landmark.occupy = [0]
            landmark.mindis = 0
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
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

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.6, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.live = 1
            agent.mindis = 0
            agent.time = 0
            agent.occupy = [0]
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
            landmark.occupy = 0
            landmark.mindis = 0
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

        return np.concatenate([np.array(time_grass)]+[np.array(agent.occupy)])



    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        main_reward = self.reward_all_in_once(agent, world)
        return main_reward

    def reward_all_in_once(self, agent, world):
        alpha = self.alpha
        num_agents = len(world.agents)
        reward_n = np.zeros(num_agents)
        # reward_n = [0]* num_agents
        # print(reward_n)
        
        alpha_sharing = self.alpha


        shape = True

        good_collide_id = []
        food_id = []

        for i, agent_new in enumerate(world.agents):
            agent_new.time += 1/26
            # collision reward:
            if agent_new.collide:
                for j, good in enumerate(world.agents):
                    if good is not agent_new:
                        if self.is_collision(good, agent_new):
                            reward_n[i] = reward_n[i]-3/num_agents*(1-alpha)
                            reward_buffer = -3/num_agents*(alpha)*np.ones(num_agents)
                            reward_n = reward_n+reward_buffer
        
        # shape food
        full_house = []
        for food in world.food:
            if food.occupy==1:
                full_house.append(1)

        if not self.no_wheel:
            for i, agent_new in enumerate(world.agents):
                min_dis = min([np.sqrt(np.sum(np.square(food.state.p_pos - agent_new.state.p_pos))) for food in world.food])
                dis_change = -1*min_dis
                agent_new.mindis = min_dis
                reward_n[i] = reward_n[i]+dis_change
        
        occupy_list = []
        mark=0


        for food in world.food:
            mark=0
            for agent_new in world.agents:
                if self.is_collision(food, agent_new):
                    mark=1
                    occupy_list.append(1)
                    reward_buffer = 6/num_agents*np.ones(num_agents)
                    reward_n = reward_n+reward_buffer
                    food.occupy=1
                    break
            if mark==0:
                food.occupy=0
        if not self.no_wheel:
            if len(occupy_list)==len(world.food):
                reward_buffer = 10*np.ones(num_agents)-agent_new.time*2
                reward_n = reward_n+reward_buffer
        for agent_new in world.agents:
            agent_new.occupy = [len(occupy_list)/len(world.food)]
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
        other_time = []
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
                other_time.append([0])
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel)
                other_live.append(np.array([other.live]))
                other_time.append(np.array([other.time]))
        result = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.live])] + entity_pos + other_pos + other_vel + other_live)
        return result

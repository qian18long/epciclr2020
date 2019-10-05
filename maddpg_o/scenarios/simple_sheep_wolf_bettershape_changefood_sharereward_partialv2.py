import numpy as np
from mpe_local.multiagent.core import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import os

SIGHT = 0.5
ALPHA = 0
class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, sight, no_wheel, ratio, food_ratio=None):
        self.n_good = n_good
        self.n_adv = n_adv
        self.n_landmarks = n_landmarks
        self.n_food = n_food
        self.n_forests = n_forests
        self.alpha = alpha
        self.sight = sight
        self.no_wheel = no_wheel
        self.size_food = food_ratio or ratio
        self.size = ratio
        # print(sight,"sight___wolf_sheep_v2")
        # print(alpha,"alpha######################")

    def make_world(self):
        world = World()
        # set any world properties first
        world.collaborative = True
        world.dim_c = 2
        world.sight = self.sight
        world.size = self.size
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
            agent.size = (0.075 if agent.adversary else 0.05)
            agent.accel = (2.0 if agent.adversary else 4.0)
            if agent.adversary:
                agent.showmore = np.zeros(num_good_agents)
            else:
                agent.showmore = np.zeros(num_food)
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = (2 if agent.adversary else 3)
            agent.live = 1
        # add landmarks
        '''
        world.walls = [Wall() for i in range(4)]
        ori = ['H','H','V','V']
        X = [0.9,-0.9,0.9,-0.9]
        Y = [1,-1,1,-1]
        for i, wall in enumerate(world.walls):
            wall.orient = ori[i]
            wall.axis_pos = Y[i]
            wall.endpoints = np.array((-1,1))
            wall.width = 0.2
            wall.color = np.array([0.25, 0.6, 0.6])
            wall.hard = True
        '''

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
        # random properties for agents
        #########

        seed = int.from_bytes(os.urandom(4), byteorder='little')
        # print("reseed to", seed)
        np.random.seed(seed)

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.45, 0.95]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
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
            landmark.color = np.array([0.15, 0.65, 0.15])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1 * self.size, +1* self.size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9* self.size, +0.9* self.size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9* self.size_food, +0.9* self.size_food, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9* self.size, +0.9* self.size, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        #########
        '''
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        index = np.random.randint(2)
        '''
        '''
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                if (i==index):
                    p_pos_rem = landmark.state.p_pos
        '''
        '''
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            # if (i==index)

        # set random initial states
        for agent in world.agents:
        '''
        '''
            key_sheep = np.random.randint(2)
            if not (agent.adversary or key_sheep):
                # index = np.random.randint(self.num_landmarks)
                agent.state.p_pos = p_pos_rem+np.random.uniform(-0.1, +0.1, world.dim_p)
            else:
                agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
        '''
        '''
            agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        '''


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

    '''
    def done(self, agent, world):

        if agent.collide:
            if agent in self.adversaries(world):
                for ag in self.good_agents(world):
                    if self.is_collision(ag, agent):
                        return 1
            else:
                for a in self.adversaries(world):
                    if self.is_collision(a, agent):
                        return 1
        return 0
    '''

    def done(self, agent, world):
        if agent in self.adversaries(world):
            for ag in self.good_agents(world):
                if ag.live:
                    return 0
            return 1
        else:
            if not agent.live:
                return 1
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
        reward_n = [0]* num_agents
        # print(reward_n)
        
        alpha_sharing = self.alpha

        agents_live_adv = []
        agents_live_good = []
        agents_live_id = []

        #reward_live_adv = []
        #reward_live_good = []
        for i, agent_new in enumerate(world.agents):
            if agent_new.live:
                agents_live_id.append(i)
                if agent_new.adversary:
                    agents_live_adv.append(agent_new)
                    #reward_live_adv.append(0)
                else:
                    agents_live_good.append(agent_new)
                    #reward_live_good.append(0)
            else:
                agent_new.color = np.array([0.0, 0.0, 0.0])
        reward_live_adv = np.zeros(len(agents_live_adv))
        reward_live_good = np.zeros(len(agents_live_good))

        shape_sheep = False
        shape_wolf = True

        good_collide_id = []
        adv_collide_id = []
        food_id = []

        for i, agent_new in enumerate(agents_live_adv):

            # collision reward:
            good_collide_id = []
            num_collide = 0
            if agent_new.collide:
                for j, good in enumerate(agents_live_good):
                    if self.is_collision(good, agent_new):
                        reward_live_adv[i] += 5*(1-alpha_sharing)
                        reward_live_good[j] -= 5*(1-alpha_sharing)
                        good.live = 0
                        adv_share_reward = np.ones(len(agents_live_adv))*5*alpha_sharing
                        good_share_reward = -np.ones(len(agents_live_good))*5*alpha_sharing
                        reward_live_adv = reward_live_adv+adv_share_reward
                        reward_live_good = reward_live_good+good_share_reward
            # shape to encourage collision:
            if len(agents_live_good)>0:
                if shape_wolf:
                    distance_min = min(np.sqrt(np.sum(np.square(agent_new.state.p_pos - good.state.p_pos)))for good in agents_live_good)
                    if distance_min<self.sight and not self.no_wheel:
                        reward_live_adv[i] -= 0.1* distance_min

        for i, agent_new in enumerate(agents_live_good):
            # shape of food:
            distance_min = min([np.sqrt(np.sum(np.square(food.state.p_pos - agent_new.state.p_pos))) for food in world.food])
            if distance_min<self.sight and not self.no_wheel:
                reward_live_good[i] -= 0.1 * distance_min

            # shape to not encourage collision:
            if shape_sheep:
                distance_min = min(np.sqrt(np.sum(np.square(agent_new.state.p_pos - adv.state.p_pos))) for adv in agents_live_adv)
                if distance_min<self.sight and not self.no_wheel:
                    reward_live_good[i] += 0.05*distance_min

            # eat food reward
            for i_food, food in enumerate(world.food):
                if self.is_collision(agent_new, food):
                    reward_live_good[i] += 2*(1-alpha_sharing)
                    good_share_reward = 2*np.ones(len(agents_live_good))*alpha_sharing
                    reward_live_good = reward_live_good+good_share_reward
                    food_id.append(i_food)

        reward_all_live = np.append(reward_live_adv, reward_live_good)
        j = 0
        for id in agents_live_id:
            reward_n[id] = reward_all_live[j]
            j = j+1
        for i_food in food_id:
            world.food[i_food].state.p_pos = np.random.uniform(-0.9 * self.size_food, +0.9 * self.size_food, world.dim_p)
        return reward_n


    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        agent.showmore = np.zeros(world.num_adversaries)
        shape = False
        if not agent.live:
            return 0
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.05 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        i_showmore = 0
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5
                    agent.showmore[i_showmore] += 1

                    agent.live = 0      #gaidong
                i_showmore += 1
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return 2.5
            # return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)
        '''
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            if x >= 1:
                agent.live = 0
            rew -= 2 * bound(x)
        '''
        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2
                food.state.p_pos = np.random.uniform(-0.9 * self.size, +0.9 * self.size, world.dim_p)
        rew -= 0.1 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        agent.showmore = np.zeros(world.num_good_agents)
        l1 = []
        if shape:
            for a in agents:
                if a.live:
                    l1.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
            if not (len(l1) == 0):
                rew -= 0.1 * min(l1)
        i_showmore = 0
        if agent.collide:
            for ag in agents:
                if ag.live:
                    # for adv in adversaries:
                    if self.is_collision(ag, agent):
                        rew += 5
                        agent.showmore[i_showmore] += 1
                i_showmore += 1
        return rew

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

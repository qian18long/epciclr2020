import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

# other_sheep and other_wolf defines how many other species agents you want to observe
OTHER_SHEEP = 1
OTHER_WOLF = 1
class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        world = World(**kwargs)
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 2
        world.num_good_agents = num_good_agents
        world.num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        num_food = 4
        num_forests = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 2.0 if agent.adversary else 4.0
            # agent.showmore marks times of agent eaton by or eat other agents
            if agent.adversary:
                agent.showmore = np.zeros(num_good_agents)
            else:
                agent.showmore = np.zeros(num_food)
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 2 if agent.adversary else 3
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


    def knn(self, agent, world):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        i_agent = 0
        i_adv = 0
        goodagent_dis = []
        adv_dis = []
        goodagent_ind = []
        adv_ind = []

        for goodagent in agents:
            if goodagent is agent:
                i_agent += 1
                continue
            goodagent_ind.append(i_agent)
            i_agent += 1
            delta_pos = agent.state.p_pos - goodagent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            goodagent_dis.append(dist)

        for adv in adversaries:
            if adv is agent:
                i_adv += 1
                continue
            adv_ind.append(i_adv)
            i_adv += 1
            delta_pos = agent.state.p_pos - adv.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            adv_dis.append(dist)
        goodsort = sorted(range(len(goodagent_dis)), key=lambda k: goodagent_dis[k])
        advsort = sorted(range(len(adv_dis)), key=lambda k: adv_dis[k])
        good_result = []
        adv_result = []
        for i in goodsort:
            good_result.append(goodagent_ind[i])
        for i in advsort:
            adv_result.append(adv_ind[i])
        return good_result, adv_result

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
        # return agent's time on grass, time of alive, times eaton by or eat other agents
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

        return np.concatenate([np.array(time_grass)]+[np.array(time_live)]+[agent.showmore])



    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

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
                food.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
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
        other_sheep = OTHER_SHEEP
        other_wolf = OTHER_WOLF
        good_result, adv_result = self.knn(agent, world)
        index_list = []

        agents_need = []

        for i in range(other_wolf):
            index = adv_result[i]
            index_list.append(np.array([index]))
            # print(index)
            agents_need.append(world.agents[index])

        for i in range(other_sheep):
            index = good_result[i]+world.num_adversaries
            # print(index)
            index_list.append(np.array([index]))
            agents_need.append(world.agents[index])


        # get positions of all entities in this agent's reference frame
        entity_pos = []

        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        other_live = []
        for other in agents_need:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
                other_live.append(np.array([other.live]))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.live])] + entity_pos + other_pos + other_vel + other_live+index_list)


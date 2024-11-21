import numpy as np
from multiagent.core import World, Agent, Landmark, Border, Check
from multiagent.scenario import BaseScenario
import math
import random

last_dist_pursue = 0.987
last_dist_determinate = 0.282
class Scenario(BaseScenario):
    def make_world(self):
        # global index
        # index = 0
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 3
        num_check = 1
        num_borders = 80  # (20 * 4) 4条边，每边20个border
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):  # agent的设置
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.025 if agent.adversary else 0.025
            agent.accel = 0.85 if agent.adversary else 1.0  # 加速度
            # agent.accel = 1
            # agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 0.28 if agent.adversary else 0.25
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        world.check = [Check() for i in range(num_check)]
        for i, check in enumerate(world.check):
            check.name = 'checkpoint %d' % i
            check.collide = False
            check.movable = False
            check.size = 0.05
            check.boundary = False
            check.shape = [[-0.025, -0.025], [0.025, -0.025],
                           [0.025, 0.025], [-0.025, 0.025]]
        # 加入 borders
        world.borders = [Border() for i in range(num_borders)]
        for i, border in enumerate(world.borders):
            border.name = 'border %d' % i
            border.collide = True
            border.movable = False
            border.size = 0.15  # 边界大小
            border.boundary = True
            # 改变边界厚度border.shape
            border.shape = [[-0.05, -0.05], [0.05, -0.05],
                            [0.05, 0.05], [-0.05, 0.05]]

            #print(border.pos)
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, agent_pos=None, check_pos=None, obstacles=None):
        # random properties for agents
        for i, agent in enumerate(world.agents):  # agent颜色
            agent.color = np.array(
                [0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])  # landmark 颜色
        # # random properties for borders
        for i, border in enumerate(world.borders):
            border.color = np.array([0.8, 0.4, 0.4])  # 边界颜色
        # # set random initial
        for i, check in enumerate(world.check):
            check.color = np.array([0.8, 0.6, 0.8])
        for agent in world.agents:
            # agent初始位置状态 [x,y]=[0,0]是在视野中心
            # agent.state.p_pos = np.random.uniform(-0.28, -0.05, world.dim_p) if agent.adversary else np.random.uniform(0.05, 0.28, world.dim_p)
            if agent_pos is None:
                # agent.state.p_pos = np.array(
                #     [random.uniform(-0.9, 0.9) for _ in range(2)]) if agent.adversary else np.array([random.uniform(-0.9, 0.9) for _ in range(2)])
                agent.state.p_pos = np.array(
                    [0.0, 0.0]) if agent.adversary else np.array([0.5, -0.5])
            else:
                agent.state.p_pos = agent_pos[0] if agent.adversary else agent_pos[1] 
            agent.state.p_vel = np.zeros(world.dim_p)  # agent初始速度
            agent.state.c = np.zeros(world.dim_c)  # agent初始交流状态
        if obstacles is None:
            pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]] # test point
        else:
            pos = obstacles
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = pos[i]  # landmark初始位置
                landmark.state.p_vel = np.zeros(world.dim_p)
        # world.check[0].state.p_pos = [np.random.uniform(-0.6,0.6),np.random.uniform(-0.6,0.6)]
        if check_pos is None:
            # world.check[0].state.p_pos = [random.uniform(-0.9, 0.9) for _ in range(2)]
            world.check[0].state.p_pos = [-0.7, 0.7]
        else:
            world.check[0].state.p_pos = check_pos
        world.check[0].state.p_vel = np.zeros(world.dim_p)
        # # 增加部分 [x,y]=[0,0]是在视野中心
        # # 每条边20个border， 计算好大概位置，依次为每条边的border生成位置坐标
        pos = []
        x = -0.95
        y = -1.0
        # bottom
        for count in range(20):
            pos.append([x, y])
            x += 0.1

        x = 1.0
        y = -0.95
        # right
        for count in range(20):
            pos.append([x, y])
            y += 0.1

        x = 0.95
        y = 1.0
        # top
        for count in range(20):
            pos.append([x, y])
            x -= 0.1

        x = -1.0
        y = 0.95
        # left
        for count in range(20):
            pos.append([x, y])
            y -= 0.1

        for i, border in enumerate(world.borders):
            border.state.p_pos = np.asarray(pos[i])  # 将设好的坐标传到border的位置坐标
            border.state.p_vel = np.zeros(world.dim_p)

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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        main_reward = self.adversary_reward(
            agent, world) if agent.adversary else self.agent_reward(agent, world)
        # main_reward = self.adversary_reward(agent, world) + self.agent_reward(agent, world)
        # main_reward = self.my_reward(agent, world)
        return main_reward

    def is_done(self, agent, world):
        # agent
        if agent.adversary:
            good_agent = self.good_agents(world)[0]
            if self.is_collision(good_agent, agent):
                return True

            
        #landmark
        if not agent.adversary:
            for i, landmark in enumerate(world.landmarks):
                delta_dis = agent.state.p_pos - landmark.state.p_pos 
                dist = np.sqrt(np.sum(np.square(delta_dis)))
                # dist_min = agent.size + landmark.size
                dist_min = 0.075
                if dist <= dist_min :
                    return True
            #check success
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos)))
            if dist < agent.size + world.check[0].size:
                return True
            
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        global last_dist_determinate, last_dist_pursue
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # 碰撞惩罚
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 40
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                if self.is_collision(landmark, agent):
                    rew -= 30
        for i, border in enumerate(world.borders):
            if self.is_collision(border, agent):
                rew -= 10

        dist = np.sqrt(
            np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos)))
        # 距离check点越远，惩罚越大
        rew -= 5 * dist
        rew -= 10 * (dist - last_dist_determinate)
        if dist < agent.size + world.check[0].size:
            # 完成任务的奖励
            rew += 1000
        for adv in adversaries:#0.1
            dist2 = min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents])
            # rew += 2 * dist2
            rew += 10 * (dist2 - last_dist_pursue)
        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        global last_dist_pursue
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # reward can optionally be shaped (decreased reward for increased distance from agents)
        if shape:
            for adv in adversaries:
                dist = min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                        for a in agents])
                rew -= 0.1 * dist
                last_dist_pursue = dist
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew
    
    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by adversaries
    #     # rew = 0
    #     # adversaries = self.adversaries(world)
    #     # # 碰撞惩罚
    #     # if agent.collide:
    #     #     for a in adversaries:
    #     #         if self.is_collision(a, agent):
    #     #             rew -= 10
    #     # for i, landmark in enumerate(world.landmarks):
    #     #     if not landmark.boundary:
    #     #         if self.is_collision(landmark, agent):
    #     #             rew -= 10
    #     # for i, border in enumerate(world.borders):
    #     #     if self.is_collision(border, agent):
    #     #         rew -= 10

    #     # dist = np.sqrt(
    #     #     np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos)))
    #     # # 距离check点越远，惩罚越大
    #     # rew -= 0.5 * dist
    #     # if dist < agent.size + world.check[0].size:
    #     #     # 完成任务的奖励
    #     #     rew += 12
    #     global last_dist_determinate
    #     rew = 0
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     # 碰撞惩罚
    #     if agent.collide:
    #         for a in adversaries:
    #             if self.is_collision(a, agent):
    #                 rew -= 100
    #     for i, landmark in enumerate(world.landmarks):
    #         if not landmark.boundary:
    #             if self.is_collision(landmark, agent):
    #                 rew -= 40
    #     for i, border in enumerate(world.borders):
    #         if self.is_collision(border, agent):
    #             rew -= 10
    #     dist = np.sqrt(
    #         np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos)))
    #     # 距离check点越远，惩罚越大
    #     rew -= 10 * (dist - last_dist_determinate)
    #     # print("agent  ", dist)
    #     rew -= 0.2 * dist
    #     for i, landmark in enumerate(world.landmarks):
    #         delta_pos = landmark.state.p_pos - agent.state.p_pos
    #         dist1 = np.sqrt(np.sum(np.square(delta_pos)))
    #         # print(dist1)
    #         if dist1 < 0.12:
    #             rew -= 10
    #     if dist < agent.size + world.check[0].size:
    #         # 完成任务的奖励
    #         rew += 5000
    #     if dist > 0.5 and dist < 1:
    #         # 完成任务的奖励
    #         rew += 1
    #     if dist > 0.2 and dist < 0.5:
    #         # 完成任务的奖励
    #         rew += 5
    #     if dist > 0.1 and dist < 0.2:
    #         # 完成任务的奖励
    #         rew += 15
    #     if dist < 0.1:
    #         # 完成任务的奖励
    #         rew += 25
    #     if 0.4 < dist < 0.8:
    #         rew += 40 * (0.8 - dist)
    #     if dist < 0.4:
    #         rew += 60 * (0.4 - dist)
    #     for adv in adversaries:#0.1
    #         dist2 = min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
    #                 for a in agents])
    #         rew += 2 * dist2
    #         # rew += 100 * (dist2 - last_dist_pursue)
    #     last_dist_determinate = dist
    #     # rew -= dist 
    #     return rew

    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     global last_dist_pursue
    #     rew = 0
    #     shape = True
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     # reward can optionally be shaped (decreased reward for increased distance from agents)
    #     for adv in adversaries:#0.1
    #         # rew += 20 * \
    #         #     min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
    #         #         for a in agents])
    #         dist = min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
    #                 for a in agents])
    #         # print("adverse  ", dist)
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     # rew += 10
    #                     rew -= 10
    #     rew += 10 * (dist - last_dist_determinate)
    #     if dist < 0.08:
    #         rew -= 0.1
    #     last_dist_pursue = dist
    #     return rew
    
    def my_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # reward can optionally be shaped (decreased reward for increased distance from agents)
        for adv in adversaries:#0.1
            rew += 20 * \
                min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        # rew += 10
                        rew -= 10

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        check_pos = []
        check_pos.append(agent.state.p_pos - world.check[0].state.p_pos)
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            #     other_vel.append(other.state.p_vel)
            other_vel.append(other.state.p_vel)  # 增加
            dists = np.sqrt(np.sum(np.square(agent.state.p_pos - other_pos)))
        # return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + dists)   # 知道自己的位置、速度和与其他agent的距离
        return np.concatenate([agent.state.p_pos] + other_pos + check_pos + entity_pos + [agent.state.p_vel] + other_vel)

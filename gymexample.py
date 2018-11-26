import gym
import numpy as np
import random
import pickle

slow = True
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make("MountainCar-v0")
observation = env.reset()


class Sarsa(object):
    def __init__(self):
        self.states = np.zeros((180, 140), dtype=float)
        self.actions = [0, 1, 2]
        self.reward = -1
        self.discount_factor = 0.95
        self.alpha = 0.99
        self.td_parameter = 0.2
        self.e_values = np.zeros((180, 140, 3), dtype=float)
        self.q_table = np.random.randn(180, 140, 3)

    def epochs(self, iters):
        for x in range(iters):
            new_observation = env.reset()

            self.learn(new_observation)
        with open("checkpoint.q", "wb") as check:
            pickle.dump(self.q_table, check)

            self.show_results()

    def show_results(self):
        print("EXECUTION")
        new_observation = env.reset()

        while True:
            env.render()
            position = int((round(new_observation[0], 2) + 1.2) * 10)
            velocity = int((round(new_observation[1], 3) + 0.07) * 100)
            a = np.argmax(self.q_table[position, velocity])
            new_observation, _, _, done = env.step(a)

            if done:
                break

    def learn(self, new_observation):
        values_set = False
        counter = 0

        while counter < 100002:
            if not values_set:
                position = int((round(new_observation[0], 2) + 1.2) * 10)
                velocity = int((round(new_observation[1], 3) + 0.07) * 100)
                a = random.randint(0, 2)

            q = self.q_table[position, velocity, a]

            # take action a
            observation_next, _, _, done = env.step(a)
            position_next = int((round(observation_next[0], 2) + 1.2)*10)
            velocity_next = int((round(observation_next[1], 3) + 0.07)*100)

            a_next = np.argmax(self.q_table[position_next, velocity_next])
            q_next = self.q_table[position_next, velocity_next, a_next]

            delta = -1 + self.discount_factor * q_next - q
            self.e_values[position, velocity, a] += 1

            self.q_table += self.alpha * delta * self.e_values
            self.e_values *= self.discount_factor * self.td_parameter
            position = position_next
            velocity = velocity_next
            a = a_next
            values_set = True
            counter += 1
            if counter % 100 == 0:
                print(counter)

            if done:
                print(counter)
                break


sar = Sarsa()
sar.epochs(5)

import gym
import time
import numpy as np

from controller import DQN


class Env(object):
    def __init__(self):
        self.env = gym.make('Breakout-v4')

        self.action = 1
        self.action_n = self.env.action_space.n
        self.controller = DQN(self.action_n)


    def _togray(self, observation):
        row, column, channel = observation.shape
        state = np.ones((row, column, 1))
        for i in range(row):
            for j in range(column):
                if (observation[i, j] == np.zeros((row, column, channel))).all():
                    state[i, j, 0] = 0
                else:
                    state[i, j, 0] = observation[i, j, 0]
        return state


    def key_press(self, key, mod):
        # print key
        if key == 65361:
            # left
            self.action = 3
        elif key == 65363:
            # right
            self.action = 2
        elif key == 65362:
            # fire
            self.action = 1

    def key_release(self, key, mod):
        pass

    def reset(self):
        self.env.reset()

    def main_loop(self, max_step=1000):
        self.env.reset()
        for _ in range(max_step):
            time.sleep(0.2)
            self.env.render()
            self.env.unwrapped.viewer.window.on_key_press = self.key_press
            self.env.unwrapped.viewer.window.on_key_release = self.key_release

            observation, reward, done, info = self.env.step(self.action) # 2 is right, 3 is left

            # game over or not
            if done:
                print 'game over'
                # game over
                self.action = 1
                self.reset()
            else:
                self.action = 0 # reset action
                print reward
                if reward:
                    print reward

    def auto_loop(self, max_episode=1000):
        step = 0
        for episode in range(max_episode):
            total_reward = 0
            observation = self.env.reset()
            observation = self._togray(observation)
            while True:
                # time.sleep(0.2)
                self.env.render()
                action = self.controller.choose_action(observation)
                print 'action: ', action

                observation_next, reward, done, info = self.env.step(action) # 2 is right, 3 is left
                observation_next = self._togray(observation_next)

                self.controller.store_data(observation, action, reward, observation_next)
                self.controller.train_network()

                observation = observation_next

                total_reward += reward

                # game over or not
                if done:
                    print 'game over'
                    break

                step += 1
            print 'Episode:{}, total_reward: {}'.format(episode, total_reward)


if __name__ == '__main__':
    game = Env()
    game.auto_loop()
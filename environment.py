import gym
import time
import numpy as np


class Env(object):
    def __init__(self):
        self.env = gym.make('Breakout-v4')
        self.observation = self.env.reset()

        self.action = 1
        self.action_n = self.env.action_space.n


    def _togray(self, observation):
        row, column, channel = observation.shape
        state = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                if not observation[i, j] == np.array((1, channel), dtype=np.uint8):
                    state[i, j] = 1
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
        for _ in range(max_step):
            time.sleep(0.2)
            self.env.render()
            self.env.unwrapped.viewer.window.on_key_press = self.key_press
            self.env.unwrapped.viewer.window.on_key_release = self.key_release

            observation, reward, done, info = self.env.step(self.action) # 2 is right, 3 is left
            # state = self._togray(observation)
            print done

            # game over or not
            if done:
                print 'game over'
                # game over
                self.action = 1
                self.reset()
            else:
                self.action = 0 # reset action
                if reward:
                    print reward


if __name__ == '__main__':
    game = Env()
    game.main_loop()
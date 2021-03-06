import gym
import time
import numpy as np
import cv2
from controller import DQN

CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
CNN_INPUT_DEPTH = 1
SERIES_LENGTH = 4

class Env(object):
    def __init__(self):
        self.env = gym.make('Breakout-v4')

        self.action = 1
        self.action_n = self.env.action_space.n
        self.controller = DQN(self.action_n)

    def _togray(self, observation):
        height, width, nchannel = observation.shape

        sHeight = int(height * 0.5)
        sWidth = CNN_INPUT_WIDTH

        state_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)

        state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

        cnn_inputImg = state_binarySmall[25:, :]
        # rstArray = state_graySmall.reshape(sWidth * sHeight)
        cnn_inputImg = cnn_inputImg.reshape((CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT))

        return cnn_inputImg

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

    def auto_loop(self, max_episode=1000000):
        step = 0
        for episode in range(max_episode):
            total_reward = 0
            observation = self.env.reset()
            observation = self._togray(observation)

            while True:
                # time.sleep(0.2)
                self.env.render() 
                state = np.stack((observation, observation, observation, observation), axis=2)
                action = self.controller.choose_action(state)
                print 'action: ', action

                observation_next, reward, done, info = self.env.step(action) # 2 is right, 3 is left

                observation_next = self._togray(observation_next)
                next_state = np.append(observation_next.reshape((80, 80, 1)), state[:,:,:3], axis= 2)

                self.controller.store_data(state, action, reward, next_state)
                self.controller.train_network()

                state = next_state

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
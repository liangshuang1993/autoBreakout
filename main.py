from environment import *
from controller import *

game = Env()

Controller = DQN(game.action_n)

game.auto_loop()

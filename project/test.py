import rospkg
import sys
PATH = rospkg.RosPack().get_path("sim2real") + "/project/TEAM1"
sys.path.append(PATH)
import gym

import env
env = gym.make('RCCar-v0')
env.seed(1)
env = env.unwrapped
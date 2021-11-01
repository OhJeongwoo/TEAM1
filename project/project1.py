#!/usr/bin/env python2
from __future__ import print_function


##### add python path #####
import sys
import os
import rospkg
import rospy

PATH = rospkg.RosPack().get_path("sim2real") + "/scripts"
print(PATH)
sys.path.append(PATH)


import gym
import env
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import json
import random
import math
import yaml
import time
from sim2real.msg import Result, Query


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval.yaml"

TEAM_NAME = "TEAM1"

def dist(waypoint, pos):
    return math.sqrt((waypoint[0] - pos.x) ** 2 + (waypoint[1] - pos.y) ** 2)

class PurePursuit:
    def __init__(self, args):
        rospy.init_node('purepursuit_' + TEAM_NAME, anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        self.track_list = self.env.track_list
        print("debug3")
        self.time_limit = 100.0
        self.lookahead = 10
        self.prv_err = None
        self.cur_err = None
        self.dt = 0.1
        self.min_vel = 1.5
        self.max_vel = 2.5
        self.Kp = 3.0
        self.Ki = 0.0
        self.Kd = 0.5

        self.query_sub = rospy.Subscriber("/query", Query, self.callback_query)
        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)

    def callback_query(self, data):
        """
        result contains
        int32 id
        int32 trial
        string team
        string world
        float32 elapsed_time
        int32 waypoints
        int32 n_waypoints
        bool success
        string fail_type
        """
        rt = Result()
        START_TIME = time.time()
        is_exit = data.exit
        print("exit: ", is_exit)
        try:
            # if query is valid, start
            if data.name != TEAM_NAME:
                return
            
            if data.world not in self.track_list:
                END_TIME = time.time()
                rt.id = data.id
                rt.trial = data.trial
                rt.team = data.name
                rt.world = data.world
                rt.elapsed_time = END_TIME - START_TIME
                rt.waypoints = 0
                rt.n_waypoints = 20
                rt.success = False
                rt.fail_type = "Invalid Track"
                self.rt_pub.publish(rt)
                return

            # implementation pure pursuit algorithm
            print("[%s] START TO EVALUATE! MAP NAME: %s" %(data.name, data.world))
            self.env.reset(name = data.world)
            self.waypoints = self.env.waypoints_list[self.env.track_id]
            self.N = len(self.waypoints)
            self.prv_err = None
            self.cur_err = None

            while True:
                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = self.env.next_checkpoints
                    rt.n_waypoints = 20
                    rt.success = False
                    rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    print("EXCEED TIME LIMIT")
                    break
                
                cur_pos = self.env.get_pose()
                d = [dist(self.waypoints[i], cur_pos.position) for i in range(self.N)]
                cur_waypoint = d.index(min(d))
                siny_cosp = 2 * (cur_pos.orientation.w * cur_pos.orientation.z + cur_pos.orientation.x * cur_pos.orientation.y)
                cosy_cosp = 1 - 2 * (cur_pos.orientation.y * cur_pos.orientation.y + cur_pos.orientation.z * cur_pos.orientation.z)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                
                dy = self.waypoints[(cur_waypoint + self.lookahead) % self.N][1] - cur_pos.position.y
                dx = self.waypoints[(cur_waypoint + self.lookahead) % self.N][0] - cur_pos.position.x
                theta = np.arctan2(dy, dx)
                self.prv_err = self.cur_err
                self.cur_err = yaw - theta

                while(self.cur_err > np.pi):
                    self.cur_err -= 2 * np.pi
                while(self.cur_err < -np.pi):
                    self.cur_err += 2 * np.pi

                input_steering = self.cur_err * self.Kp
                if self.prv_err is not None:
                    input_steering += (self.cur_err - self.prv_err) / self.dt * self.Kd
                
                input_vel = np.clip(self.max_vel / (1 + abs(input_steering)), self.min_vel, self.max_vel)
                _, _, done, logs = self.env.step([input_steering, input_vel])

                if done:
                    END_TIME = time.time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = logs['checkpoints']
                    rt.n_waypoints = 20
                    rt.success = True if logs['info'] == 3 else False
                    rt.fail_type = ""
                    print(logs)
                    if logs['info'] == 1:
                        rt.fail_type = "Collision"
                    if logs['info'] == 2:
                        rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    print("publish result")
                    break
        except:
            END_TIME = time.time()
            rt.id = data.id
            rt.trial = data.trial
            rt.team = data.name
            rt.world = data.world
            rt.elapsed_time = END_TIME - START_TIME
            rt.waypoints = 0
            rt.n_waypoints = 20
            rt.success = False
            rt.fail_type = "Script Error"
            self.rt_pub.publish(rt)

        if is_exit:
            rospy.signal_shutdown("End query")
        
        return

if __name__ == '__main__':
    with open(yaml_file) as file:
        args = yaml.load(file)
    PurePursuit(args)
    rospy.spin()


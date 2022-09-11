#!/usr/bin/python
# -*- coding: utf8 -*- 

"""Base actors on which residuals are learned."""
import numpy as np
import random as rnd
import time, pygame
import gin

#####################################
# Change these to match your joystick
RIGHT_UP_AXIS = 4
RIGHT_SIDE_AXIS = 3
LEFT_UP_AXIS = 1
LEFT_SIDE_AXIS = 0
#####################################

@gin.configurable
class UR10JoystickActor(object):
    """Joystick Controller for UR10."""

    def __init__(self, action_mask=[1, 1, 1, 1, 1, 1], random=False):
        """Init."""
        self.action_mask = action_mask
        self.random = random
        self.rnd = rnd
        self.rnd.seed(0)
        self.human_agent_action = np.array([[0., 0.], [0., 0.]], dtype=np.float32)  # noop
        self.button = np.array([0], dtype=np.int32)
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x)
                     for x in range(pygame.joystick.get_count())]
        if len(joysticks) > 1:
            raise ValueError("There must be exactly 1 joystick connected.",
                             "Found ", len(joysticks))
        elif len(joysticks) == 0:
            raise ValueError("There is no joystick connected.")
        elif len(joysticks) == 1:   
            self.joy = joysticks[0]
            self.joy.init()
        pygame.init()
        self.t = None
        self.action_period = 10
        self.action_cnt = self.action_period
        

    def _get_human_action(self):
        for event in pygame.event.get():
            # Joystick input
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == LEFT_SIDE_AXIS:
                    self.human_agent_action[0, 1] = event.value
                elif event.axis == LEFT_UP_AXIS:
                    self.human_agent_action[0, 0] = -1.0 * event.value
                if event.axis == RIGHT_SIDE_AXIS:
                    self.human_agent_action[1, 1] = event.value
                elif event.axis == RIGHT_UP_AXIS:
                    self.human_agent_action[1, 0] = -1.0 * event.value
            if event.type == pygame.JOYBUTTONDOWN:
                self.button[0] = event.button
            else: # button clear
                self.button[0] = -1
                
        if abs(self.human_agent_action[0, 0]) < 0.01:
            self.human_agent_action[0, 0] = 0.0
        if abs(self.human_agent_action[1, 0]) < 0.01:
            self.human_agent_action[1, 0] = 0.0
        action = np.array([self.human_agent_action[0][0], 
                            self.human_agent_action[0][1], 
                            self.human_agent_action[1][0], 
                            0, 
                            0, 
                            self.human_agent_action[1][1], 
                            self.human_agent_action[0]])
        return action
    
    def _get_random_action(self):
        self.action_cnt += 1
        if self.action_cnt > self.action_period:
            print(self.action_mask)
            self.action = np.array([self.rnd.choice([-1, 0, 1])*self.action_mask[0], 
                                    self.rnd.choice([-1, 0, 1])*self.action_mask[1], 
                                    self.rnd.choice([-1, 0, 1])*self.action_mask[2], 
                                    self.rnd.choice([-1, 0, 1])*self.action_mask[3], 
                                    self.rnd.choice([-1, 0, 1])*self.action_mask[4], 
                                    self.rnd.choice([-1, 0, 1])*self.action_mask[5], 
                                    -1.0])
            self.action_cnt = 0
            self.action_period = rnd.randrange(5,101)
            
        return self.action

    def __call__(self, ob):
        """Act."""
        if self.random:
            action = self._get_random_action()
        else:
            action = self._get_human_action()
        self.t = time.time()
        return action

    def reset(self):
        self.human_agent_action[:] = 0.


if __name__ == '__main__':
    import gym
    from dl.rl import ensure_vec_env
    import time

    env = gym.make("ur10_env:ur10-v0")
    #env = ensure_vec_env(env)

    actor = UR10JoystickActor(random=True)

    for _ in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0
        time.sleep(1.)

        while not done:
            ob, r, done, _ = env.step(actor(ob))
            env.render()
            reward += r
        print(reward)

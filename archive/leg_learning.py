import numpy as np
import random
import math
from time import sleep

import leg.dynamics as leg

class ActionSpace:
	def __init__(self, actions):
		self.actions = actions
		self.n = len(actions)

	def sample(self):
		return np.random.randint(self.n)

class ObservationSpace:
	def __init__(self, low, high, mode):
		self.low = low
		self.high = high
		self.mode = mode
		self.shape = (4,)

	def sample(self):
		#q = []
		#for l, h, m in zip(self.low, self.high, self.mode):
		#	q.append(np.random.triangular(l, m, h))
		#return q
		return np.array(self.mode)

class LegEnvironment:
	def __init__(self, dt):
		self.dt = dt
		self.action_space = ActionSpace([-0.1, -0.001, 0, 0.001, 0.1])
		self.observation_space = ObservationSpace([-1.0, 3.0, -2.0, 1.0, -2.0, -3.0, -1.0, -1.0],
			[1.0, 10.0, 2.0, 3.0, 2.0, 3.0, 1.0, 1.0],
			[0.0, 7.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

	def getState(self):
		#state = []
		#for qi, (l, h) in zip(self.robot.q, zip(self.observation_space.low, self.observation_space.high)):
		#	state.append(np.clip(qi, l, h))
		#return np.array(state)
		y = self.robot.q[1]
		dy = self.robot.q[5]
		theta = np.arctan2(self.robot.q[2], self.robot.q[3])
		dtheta = (self.robot.q[6] * self.robot.q[3] - self.robot.q[7] * self.robot.q[2]) / (self.robot.q[2] ** 2 + self.robot.q[3] ** 2)
		dx = self.robot.q[4]
		return np.array([y, theta, dtheta, dx])

	def reset(self):
		q0 = self.observation_space.sample()
		self.robot = leg.LegSystem(q0, 0.0, self.dt)
		while not self.isOK():
			q0 = self.observation_space.sample()
			self.robot = leg.LegSystem(q0, 0.0, self.dt)
		self.qs = [q0]
		self.ts = [0.0]
		return self.getState()

	def reset_test(self):
		q0 = np.array(self.observation_space.mode)
		self.robot = leg.LegSystem(q0, 0.0, self.dt)
		self.qs = [q0]
		self.ts = [0.0]
		return self.getState()

	def step(self, action):
		prevx = self.robot.q[0]
		self.robot.step(self.action_space.actions[action])
		self.qs.append(self.robot.q)
		self.ts.append(self.robot.t)

		obv = self.getState()

		theta = np.arctan2(self.robot.q[2], self.robot.q[3])
		dx = self.robot.q[4]

		if not self.isOK():
			return obv, 0.0, True, None
		return obv, self.robot.q[0] - prevx, False, None # - 0.25 * (np.exp(dx / 2.0) / (1 + np.exp(dx / 2.0))) ** 2 - 0.25 * np.abs(theta)

	def isOK(self):
		return self.robot.isAlive() and np.abs(np.arctan2(self.robot.q[2], self.robot.q[3])) < np.pi / 2

	def video(self):
		leg.showVideo(self.qs, self.dt)


## Initialize the "Cart-Pole" environment
env = LegEnvironment(0.03)

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (7, 6, 6, 5)  # (y, theta, dtheta, dx)
# Number of discrete actions
NUM_ACTIONS = env.action_space.n
# Bounds for each discrete state
STATE_BOUNDS = list(zip([-0.75, -np.pi/2, -np.pi/2, -3.0], [4.0, np.pi/2, np.pi/2, 3.0]))
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.1
MIN_LEARNING_RATE = 0.2

## Defining the simulation related constants
NUM_EPISODES = 50000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = False

def simulate():
    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0

    for episode in range(NUM_EPISODES):
        reward_sum = 0

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)

        for t in range(MAX_T):
            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)

                print("")

            reward_sum += reward

            if done:
               print("Episode %d finished after %f time steps, %f reward" % (episode, t, reward_sum))
               if (t >= SOLVED_T):
                   num_streaks += 1
               else:
                   num_streaks = 0
               break

            #sleep(0.25)
        #env.video()

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
    return env


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def run_best():
    actions = []
    env.reset()
    state = state_to_bucket(env.observation_space.sample())
    while True:
        action = select_action(state, 0.0)
        actions.append(action)
        obv, reward, done, _ = env.step(action)
        state = state_to_bucket(obv)
        if done:
            break
    return actions, env.qs

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/10)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/10)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def run_best():
    obv = env.reset()
    reward_sum = 0
    while True:
        state = state_to_bucket(obv)
        action = select_action(state, 0.0)
        obv, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            break
    return reward_sum

if __name__ == "__main__":
    env = simulate()

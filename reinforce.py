import numpy as np
import twolink.environment as environment
import scipy.spatial.kdtree as kdtree
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

class Learner:
	def __init__(self, env, action_low, action_high, buckets=10, t_cutoff=5.0, particles=10):
		self.env = env
		self.n = env.n
		self.k = env.k
		self.t_cutoff = t_cutoff
		self.buckets = buckets
		self.Q = {}
		self.N = {}
		self.i_episode = 0
		self.action_low = action_low
		self.action_range = action_high - action_low
		self.particles = particles

	def normalize(self, states):
		return (states - self.mean) / self.std

	def denormalize(self, states):
		return self.mean + states * self.std

	def discretize(self, state):
		return tuple((self.buckets * state).astype("int"))

	def lookup_Q(self, state):
		bucket = self.discretize(state)
		if bucket in self.Q:
			return self.Q[bucket]
		else:
			return 0.0

	def store_Q(self, state, Q):
		bucket = self.discretize(state)
		if bucket in self.Q:
			n = self.N[bucket]
			self.Q[bucket] = max(Q, self.Q[bucket]) #(n * self.Q[bucket] + Q) / (n + 1)
			self.N[bucket] += 1
		else:
			self.Q[bucket] = Q
			self.N[bucket] = 1

	def random_actions(self, n):
		return np.random.laplace(size=(n, self.k)) * self.action_range.reshape(1, -1) + self.action_low.reshape(1, -1)
		#return np.random.permutation(np.array([-0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1])).reshape(-1, 1)

	def choose_action(self, state, explore=False):
		possible_actions = self.random_actions(self.particles)
		best_action = possible_actions[0]
		best_Q = 0.0
		if not explore:
			for action in possible_actions:
				next_state, reward, done = self.env.peek(action)
				Q = reward + self.lookup_Q(next_state)
				if not done:
					if Q > best_Q:
						best_action = action
						best_Q = Q
		return best_action

	def run_episode(self, epsilon=0.0):
		state = self.env.reset()
		state_history = [state]
		reward_history = []
		while True:
			action = self.choose_action(state, np.random.random() < epsilon)
			state, reward, done = self.env.step(action)
			reward_history.append(reward)
			if done or self.env.robot.t > self.t_cutoff:
				break
			state_history.append(state)
		self.i_episode += 1
		states = np.array(state_history)
		Qs = np.array(np.cumsum(reward_history[::-1])[::-1])
		for state, Q in zip(states, Qs):
			self.store_Q(state, Q)
		return state_history, reward_history

def plot_Q(learner):
	def scatter3d(x,y,z, cs, colorsMap='jet'):
	    cm = plt.get_cmap(colorsMap)
	    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
	    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
	    fig = plt.figure()
	    ax = Axes3D(fig)
	    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
	    scalarMap.set_array(cs)
	    fig.colorbar(scalarMap)
	    plt.show()
	xyz = np.array(list(learner.Q.keys()), dtype="float")
	cs = np.array(list(learner.Q.values()))
	scatter3d(xyz[:,0], xyz[:,1], xyz[:,2], cs)

dt = 0.03
tau_low = -0.1
tau_high = 0.1
exploration_rate = 0.5
exploration_time_constant = 250
episodes = 5000

if __name__ == "__main__":
	env = environment.Environment(dt)
	learner = Learner(env, tau_low * np.ones(env.k), tau_high * np.ones(env.k), particles=20)
	for i_episode in range(episodes):
		epsilon = exploration_rate / (1 + i_episode / exploration_time_constant)
		states, rewards = learner.run_episode(epsilon)
		print("Episode", i_episode)
		print("Exploration rate:", epsilon)
		print("Total reward:", np.sum(rewards))

import numpy as np
import twolink.dynamics as dynamics

class Environment:
	def __init__(self, dt):
		self.k = 2
		self.n = 3
		self.dt = dt

		self.robot = None
		self.reset()

	def getState(self, z=None):
		if z is None:
			z = self.robot.z
		y0 = z[1]
		t1 = z[2]
		t2 = z[3]
		dt1 = z[6]
		dt2 = z[7]
		return np.array([y0, t1, t2])

	def reset(self):
		z0 = np.array([0, 7, 0, 0, 0, 0, 0, 0])
		if self.robot is None:
			self.robot = dynamics.TwoLinkSystem(z0, self.dt)
		else:
			self.robot.reset(z0)
		self.zs = [z0]
		self.ts = [0.0]
		self.rewards = []
		self.actions = []
		return self.getState()

	def step(self, action):
		prevx0 = self.robot.z[0]
		self.robot.step(action)
		self.zs.append(self.robot.z)
		self.ts.append(self.robot.t)
		self.actions.append(action)

		obv = self.getState()

		x0 = self.robot.z[0]

		if not self.isOK():
			return obv, 0.0, True
		reward = x0 - prevx0
		self.rewards.append(reward)
		return obv, reward, False

	def peek(self, action):
		z = self.robot.peek(action)
		obv = self.getState(z)
		if not self.isOK(z):
			return obv, 0.0, True
		else:
			return obv, z[0] - self.robot.z[0], False

	def simulate(self, t_cutoff, controller):
		state = self.reset()
		while True:
			action = controller(state)
			state, reward, done = self.step(action)
			if done or self.robot.t > t_cutoff:
				break

	def isOK(self, z=None):
		if z is None:
			z = self.robot.z
		t1 = z[2]
		t2 = z[3]
		return self.robot.isAlive(z) and abs(t1) < np.pi / 2 and abs(t2) < np.pi / 2

	def video(self):
		dynamics.showVideo(self.robot, self.zs, self.dt)

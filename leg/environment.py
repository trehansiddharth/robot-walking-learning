import numpy as np
import leg.dynamics as dynamics

class Environment:
	def __init__(self, dt):
		self.dt = dt
		self.n = 3
		self.k = 1
		self.rewards = []
		self.actions = []

	def getState(self, q=None):
		if q is None:
			q = self.robot.q
		y = q[1]
		dy = q[5]
		sx = q[2]
		sy = q[3]
		theta = np.arctan2(sx, sy)
		dtheta = (q[6] * q[3] - q[7] * q[2]) / (q[2] ** 2 + q[3] ** 2)
		dx = q[4]
		return np.array([y, theta, dtheta])

	def reset(self):
		q0 = np.array([0, 7, 0, 2.0, 0, 0, 0, 0])
		self.robot = dynamics.LegSystem(q0, 0.0, self.dt)
		self.qs = [q0]
		self.ts = [0.0]
		return self.getState()

	def step(self, action):
		prevx = self.robot.q[0] + self.robot.q[2]
		self.robot.step(action[0])
		self.qs.append(self.robot.q)
		self.ts.append(self.robot.t)
		self.actions.append(action)

		obv = self.getState()

		x = self.robot.q[0]
		sx = self.robot.q[2]
		theta = np.arctan2(self.robot.q[2], self.robot.q[3])
		dx = self.robot.q[4]

		if not self.isOK():
			return obv, 0.0, True
		reward = x + sx - prevx
		self.rewards.append(reward)
		return obv, reward, False

	def peek(self, action):
		q = self.robot.peek(action[0])
		obv = self.getState(q)
		if not self.isOK(q):
			return obv, 0.0, True
		else:
			return obv, q[0] + q[2] - self.robot.q[0] - self.robot.q[2], False

	def simulate(self, t_cutoff, controller):
		state = self.reset()
		while True:
			action = controller(state)
			state, reward, done = self.step(action)
			if done or self.robot.t > t_cutoff:
				break

	def isOK(self, q=None):
		if q is None:
			q = self.robot.q
		return self.robot.isAlive(q) and np.abs(np.arctan2(q[2], q[3])) < np.pi / 2

	def video(self):
		dynamics.showVideo(self.qs, self.dt)

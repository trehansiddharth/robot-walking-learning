import numpy as np
import twolink.environment as environment
import pyswarm

dt = 0.03
t_cutoff = 5.0
maxiter = 100
scale = 0.01
density = 7.0

def swarm_analog(env):
    n = env.n
    k = env.k

    swarmsize = int(density ** ((n + 1) / 2))

    x_lower = -scale * np.ones(k * (n + 1))
    x_upper = scale * np.ones(k * (n + 1))

    def controller(x):
        X = x.reshape(k, n + 1)
        W = X[:,:n]
        b = X[:,n:]
        return lambda state: (np.matmul(W, state.reshape(-1, 1)) + b).reshape(-1)

    def f(x):
        env.reset()
        env.simulate(t_cutoff, controller(x))
        return -abs(np.sum(env.rewards))

    best_x, best_reward = pyswarm.pso(f, x_lower, x_upper, debug=True, maxiter=maxiter, swarmsize=swarmsize)
    return controller(best_x)

if __name__ == "__main__":
    env = environment.Environment(dt)
    best_controller = swarm_analog(env)

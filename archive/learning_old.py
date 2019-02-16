def policy_gradient(n, k):
    W = tf.get_variable("W", [n, k])
    b = tf.get_variable("b", [1, k])
    state = tf.placeholder("float", [None, n])
    actions = tf.placeholder("float", [None, k])

    linear = tf.add(tf.matmul(state, W), b)
    probabilities = tf.nn.softmax(linear)

    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), 1)
    log_probabilities = tf.log(good_probabilities)
    loss = -tf.reduce_sum(log_probabilities)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

def classifier(W, b, x):
    linear = np.matmul(x, W) + b
    softmax = (np.exp(linear) / np.sum(np.exp(linear))).reshape(-1)
    return np.argmax(softmax)

def random_search():
    env = environment.Environment(dt)
    n = env.observation_space.shape[0]
    k = env.action_space.n

    best_W = None
    best_b = None
    best_reward = -np.inf
    for i_episode in range(num_episodes):
        W = np.random.rand(n, k) * 2 - 1
        b = np.random.rand(1, k) * 2 - 1
        env = environment.Environment(dt)
        env.simulate(t_cutoff, lambda state: classifier(W, b, state))
        reward = abs(np.sum(env.rewards))
        lifetime = env.robot.t
        print("Episode:", i_episode)
        print("Lifetime: ", lifetime)
        print("Reward:", reward)
        if reward > best_reward:
            best_W = W
            best_b = b
            best_reward = reward
        print("Best reward:", best_reward)
    return lambda state: classifier(best_W, best_b, state)

def random_search_analog():
    env = environment.Environment(dt)
    n = env.observation_space.shape[0]

    scale = 10.0

    best_w = None
    best_b = None
    best_reward = -np.inf
    for i_episode in range(num_episodes):
        w = scale * (np.random.rand(n) * 2 - 1)
        b = scale * (np.random.rand() * 2 - 1)
        env = environment.Environment(dt)
        env.simulate(t_cutoff, lambda state: np.dot(w, state) + b)
        reward = abs(np.sum(env.rewards))
        lifetime = env.robot.t
        print("Episode:", i_episode)
        print("Lifetime: ", lifetime)
        print("Reward:", reward)
        if reward > best_reward and env.isOK():
            best_w = w
            best_b = b
            best_reward = reward
        print("Best reward:", best_reward)
    return lambda state: np.dot(best_w, state) + best_b

def hill_climbing():
    env = environment.Environment(dt)
    n = env.observation_space.shape[0]
    k = env.action_space.n

    exploration_rate = 0.1
    learning_rate = 0.1

    best_W = None
    best_b = None
    best_reward = -np.inf
    for i_episode in range(num_episodes):
        dW = np.random.rand(n, k) * 2 - 1
        db = np.random.rand(1, k) * 2 - 1
        if best_reward >= 0.0 and np.random.rand() > exploration_rate:
            W = best_W + learning_rate * dW
            b = best_b + learning_rate * db
        else:
            W = dW
            b = db
        env = environment.Environment(dt)
        env.simulate(t_cutoff, lambda state: classifier(W, b, state))
        reward = abs(np.sum(env.rewards))
        lifetime = env.robot.t
        print("Episode:", i_episode)
        print("Lifetime: ", lifetime)
        print("Reward:", reward)
        if reward > best_reward:
            best_W = W
            best_b = b
            best_reward = reward
        print("Best reward:", best_reward)
    return lambda state: classifier(best_W, best_b, state)

def swarm():
    env = environment.Environment(dt)
    n = env.observation_space.shape[0]
    k = env.action_space.n

    X_lower = -1 * np.ones((n + 1, k)).reshape(-1)
    X_upper = 1 * np.ones((n + 1, k)).reshape(-1)

    def f(x):
        X = x.reshape(n + 1, k)
        W = X[:n,:]
        b = X[n:,:]
        env = environment.Environment(dt)
        env.simulate(t_cutoff, lambda state: classifier(W, b, state))
        return -abs(np.sum(env.rewards))

    best_x, best_reward = pyswarm.pso(f, X_lower, X_upper, debug=True, maxiter=50)
    best_X = best_x.reshape(n + 1, k)
    best_W = best_X[:n,:]
    best_b = best_X[n:,:]
    return lambda state: classifier(best_W, best_b, state)

import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


# turns list of integers into an int
# Ex.
# build_state([1,2,3,4,5]) -> 12345
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


class FeatureTransformer:
    def __init__(self):
        # Note: to make this better you could look at how often each bin was
        # actually used while running the script.
        # It's not clear from the high/low values nor sample() what values
        # we really expect to get.
        self.cart_position_bins = np.linspace(-1.25, 0.5, 9)
        self.cart_velocity_bins = np.linspace(-0.056, 0.05,
                                              9)  # (-inf, inf) (I did not check that these were good values)


    def transform(self, observation):
        # returns an int
        cart_pos, cart_vel = observation


        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),


        ])


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = 10 ** 2
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=0, high=2, size=(num_states, num_actions))
        print(self.Q[45])

    def predict(self, s):
        x= self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        x = self.feature_transformer.transform(s)
        self.Q[x, a] += 1e-2 * (G - self.Q[x, a])

    def sample_action(self, s, eps):  # exploration exploitation balance
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)


def play_one(model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    bins=np.linspace(-1, 1, 9)


    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        action_=np.digitize(action,bins)

        prev_observation = observation
        observation, reward, done, info = env.step(action)

        totalreward += reward




        # update the model
        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        iters += 1

    return totalreward


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.56

    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        '''if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", eps)'''
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    # plot_running_avg(totalrewards)

    for i in range(20):
        eps = 1.0 / np.sqrt(i + 1)


        def final_play(env):
            done = False
            obv = env.reset()
            while not done:
                action = model.sample_action(model, eps, gamma)
                env.render(action)
                obv, reward, done, info = env.step(action)
                if done:
                    break

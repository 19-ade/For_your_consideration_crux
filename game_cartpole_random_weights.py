import gym
import numpy as np
def get_action(s,w):
    if s.dot(w)>0:
        return(1)
    else:
        return(0)
def single_action(weights, env):
    done=False
    obv=env.reset()
    t=0
    while not done :
        action=get_action(obv,weights)
        obv, reward,done,info=env.step(action)
        t=t+1
        if done:
            break
    return t
def multi_action(env, weights,t):
    episode_list=np.empty(t)
    for i in range(t):
        episode_list[i]=single_action(weights,env)


    return episode_list.mean()
def random_weights(env):
    best=0
    best_params=None

    for i in range(100):
        new_params=np.random.random(4)*2-1
        _a=multi_action(env,new_params,100)
        if _a>best:
            best=_a
            best_params=new_params
    return best_params


def final_play(env,params):
    done=False
    obv=env.reset()
    while not done:
        action=get_action(obv,params)
        env.render(action)
        obv,reward,done,info=env.step(action)
        if done:
            break


if __name__=="__main__":
    env=gym.make("CartPole-v0")
    params=random_weights(env)
    for i in range(20):
        final_play(env,params)








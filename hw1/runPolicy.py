# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:22:48 2017

@author: zhangyaqian
"""

import tensorflow as tf
import numpy as np
from BehavioralClone import build_network
import pickle

def load_params():
    params=pickle.load(open("model/params.pck",'rb'))
    #print (params.astype)
    return params['dim_obsv'],params['network_size'],params['dim_act']



def runPolicy( folder,env_name,num_rollouts):

    dim_obsv,network_size,dim_act = load_params()
    X,Y,z3,cost = build_network(dim_obsv,network_size,dim_act)
    saver = tf.train.Saver()


    with tf.Session() as sess:
        saver.restore(sess,folder)

        import gym
        env = gym.make(env_name)


        returns = []
        observations = []
        actions = []

        for i in range(num_rollouts):
            #print('iter', i)
            obs = env.reset()
            #print (obs.shape)
            done = False
            totalr = 0.
            steps = 0
            max_step = 100000# env.spec.timestep_limit
            #print (max_step)
            while not done:
                #action = policy_fn(obs[None,:])
                np_obs = np.reshape(obs,[1,-1])

                action =sess.run(z3,feed_dict={X:np_obs})

                observations.append(obs) # dim = 11
                actions.append(action) # dim = 3
                obs, r, done, _ = env.step(action)

                totalr += r
                steps += 1
                env.render()
                #if steps % 100 == 0: print("%i/%i" % (steps, max_step))
                if(steps > max_step):
                    break
            returns.append(totalr)
            env.render(close=True)
            #print("return",totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        return observations,np.mean(returns),np.std(returns)
        #np.save("data/"+env_name+"newObsv", np.array(observations))

if __name__ == "__main__":
    c#'Hopper-v1')#'''HalfCheetah-v1')
        
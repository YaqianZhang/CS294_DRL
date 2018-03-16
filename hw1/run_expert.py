#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import sys
from BehavioralClone import BC
from runPolicy import runPolicy
import matplotlib.pyplot as plt
import time

def checkExistence(vec,arr):
    N,D=arr.shape
    vec2=np.sum(np.abs(arr-vec),axis=1)
    return any(vec2==0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        #print env.action_space.high,env.action_space.low
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            #print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:


                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        folder = "data/"
        #observations=np.array(observations)
        #actions = np.array(actions)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        obsv_folder=folder+args.envname+"observations"
        action_folder = folder+args.envname+"actions"
        
        #np.save(obsv_folder,np.array(observations))
        #np.save(action_folder,np.array(actions))
    dagger_returns =[]
    dagger_returns_std=[]

    for times in range(20):
        print ("~~~~~~~~~~~~~~~~~~~~~Dagger times",times,"training datasize",len(observations),"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        time1=time.clock()
        BC(args.envname,observations,actions) ## Train using "data/envname obsv/action" ,save model in "model"
        time2= time.clock()
        print("training time: ", time2 - time1)
        print ("~~~~~~~~~~~~~~~~~~testing~~~~~~~~~~~~~")
        newObsv,mean_r,std_r = runPolicy('model/res', args.envname,5) ## load model from "model", get new observation datasets "data/env_name+ new obsvations"
        newObsv = np.array(newObsv)
        N,D=newObsv.shape
        dagger_returns.append(mean_r)
        dagger_returns_std.append(std_r)
        time3 = time.clock()

        print("testing time: ",time3-time2)

        print("~~~~~~~~~~~~~~~~~~get new data from expert~~~~~~~~~~~~~~")
        tf.reset_default_graph()
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        num_newData=0
        with tf.Session():
            tf_util.initialize()
            for i in range(N):
                alreadyHave=checkExistence(newObsv[i,:],np.array(observations))
                if(not alreadyHave):
                    num_newData += 1
                    newAction = policy_fn(newObsv[i,:][None,:])
                    actions.append(newAction)
                    observations.append(newObsv[i,:])
        print("add more training data", num_newData,N)
        time4=time.clock()
        print("aggregation time: ", time4 - time3)
    #plt.plot(dagger_returns)
    #plt.show()
    np.save("result/Dagger_return",np.array(dagger_returns))
    np.save("result/Dagger_return_std", np.array(dagger_returns_std))





if __name__ == '__main__':
    main()

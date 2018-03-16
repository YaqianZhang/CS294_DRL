# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:07:22 2017

@author: zhangyaqian
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle


def build_network(dim_obsv,network_size,dim_act):
    ## graph input
    X = tf.placeholder(tf.float32, [None, dim_obsv])
    Y = tf.placeholder(tf.float32, [None, dim_act])

    ### variable
    r1 = 0.1* np.sqrt(6./(dim_obsv+network_size[0]))
    r2= 0.1* np.sqrt(6./(network_size[0]+network_size[1]))
    r3 = 0.1 * np.sqrt(6. / (network_size[1] + network_size[2]))
    #print(r1,r2,r3)

    w1 = tf.Variable(tf.random_uniform([dim_obsv,network_size[0]],-r1,r1))
    b1= tf.Variable(tf.zeros([1,network_size[0]],tf.float32))
    w2 = tf.Variable(tf.random_uniform([network_size[0],network_size[1]],-r2,r2))
    b2= tf.Variable(tf.zeros([1,network_size[1]],tf.float32))
    w3 = tf.Variable(tf.random_uniform([network_size[1],network_size[2]],-r3,r3))
    b3= tf.Variable(tf.zeros([1,network_size[2]],tf.float32))
    
    z1=tf.nn.relu(tf.matmul(X,w1)+b1)
    z2=tf.nn.relu(tf.matmul(z1,w2)+b2)
    z3 = tf.nn.relu(tf.matmul(z2,w3)+b3)
    reg_loss=tf.reduce_sum(tf.square(w1))+tf.reduce_sum(tf.square(w2))+tf.reduce_sum(tf.square(w3))
    cost = tf.reduce_mean(tf.reduce_sum(tf.pow(z3-Y,2),1)) #+ 0.5*reg_loss
    return X,Y,z3,cost

def save_graph_params(folder,dim_obsv,network_size,dim_act):
    params={}
    params['dim_obsv']=dim_obsv
    params['network_size']=network_size
    params['dim_act']=dim_act
    #np.save(folder,params)
    pickle.dump(params,open(folder+'.pck','wc'))




def BC (env_name,obsv,action):
    ####input
    res_op = 'model/res'
    if obsv == None:
        folder="data/"
        obsvs_file=folder+env_name+'observations.npy'
        actions_file=folder+env_name+'actions.npy'


        obsv = np.load(obsvs_file)
        action = np.load(actions_file)
    else:
        obsv = np.array(obsv)
        action = np.array(action)

    
    N,dim_obsv = np.shape(obsv)
    action = np.reshape(action,(N,-1))

    N,dim_obsv=np.shape(obsv)
    N,dim_act = np.shape(action)



  
    ##parameters
    learning_rate=0.0001
    batch_size= 100
    training_iters=50
    network_size=[70,70,dim_act]
    display_step = training_iters/2
    save_graph_params('model/params',dim_obsv,network_size,dim_act)
    #print("dimensions",dim_obsv,dim_act,N)
    


    X,Y,z3,cost =build_network(dim_obsv,network_size,dim_act)
    
    optm = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    loss=[]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for j in range(training_iters):

            for i in range(N / batch_size):
                idx = np.random.randint(0,N,batch_size)
                obsv_batch = obsv[idx,]
                action_batch = action[idx,]
                l,_ = sess.run([cost,optm],feed_dict={X:obsv_batch, Y: action_batch})
                loss.append(l)
            if j % display_step ==0:
                print ("epochs %i/%i"%(j,training_iters))
                print ("loss is %f" % l)
        save_path =saver.save(sess,res_op)
        print ("Model saved in file: %s"%save_path)
        #plt.plot(loss)
        #plt.show()
    tf.reset_default_graph()




if __name__ == '__main__' :
     #BC('Hopper-v1',5)
     BC('HalfCheetah-v1',5,None,None)#observations,actions)
     #BC('Humanoid-v1',2)
 




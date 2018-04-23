import gym
import tensorflow as tf

import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
from collections import deque
import matlab
import matlab.engine
import matplotlib.pyplot as plt
cost_batch = []
Rr_buffer = []
y_rec_batch = []
#Rr_noe_buffer = []
#sess = tf.Session()
eng = matlab.engine.start_matlab()
#eng.workspace['Rr'] = Rr_real
eng.workspace['id'] = 40.0
eng.workspace['iq'] = 50.0
eng.workspace['del_Rr'] = 0.0
eng.workspace['Rr'] = 0.2
Rr = 0.2
#Rrr = 0.2
# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.3 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 2000 # experience replay buffer size
BATCH_SIZE = 32# size of minibatch


class DQN():
  # DQN Agent
  #def __init__(self, env):
  def __init__(self):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    #self.state_dim = env.observation_space.shape[0]
    #self.action_dim = env.action_space.n
    
    self.state_dim = 6
    self.action_dim = 21

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.initialize_all_variables())

  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    self.Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - self.Q_action))
    
    
    #print("Q_action is " + str(Q_action))
    #print("cost is " + str(self.cost))
    #self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

  #def perceive(self,state,action,reward,next_state,done):
  def perceive(self,state,action,reward,next_state):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    #print('minibatch = '+str(minibatch))
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    #cost_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
     # done = minibatch[i][4]
      #if done:
        #y_batch.append(reward_batch[i])
      #else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        #print("y_batch is " + str(y_batch))
        #print("reward[i] is " + str(reward_batch[i]))
        #print("Q_value_batch is " + str(Q_value_batch))
    #print("y_batch is " + str(y_batch))
    y_rec_batch.append(y_batch)
    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })
    cost = self.cost.eval(feed_dict={
      self.action_input:action_batch,
      self.y_input:y_batch,
      self.state_input:state_batch
      })
    cost_batch.append(cost)
    print("cost is " + str(cost))
    Q_rel = self.Q_action.eval(feed_dict={
      self.action_input:action_batch,
      #self.y_input:y_batch,
      self.state_input:state_batch
      })
    #Q_rel_buffer.append(Q_rel)
    print("Q_action is " + str(Q_rel))
  def egreedy_action(self,state):
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    print("Q_value is " + str(Q_value))
    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/1000
    print(" epsilon is "  + str(self.epsilon))
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)
    else:
      return np.argmax(Q_value)

    

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)*0.01
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)*0.01
    return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters

EPISODE = 3000 # Episode limitation
STEP = 100# Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  #env = gym.make(ENV_NAME)
  agent = DQN()
  #state = np.array([0.0,0.0])
    # Train
  
  tor_max = 200
  counter = 0
 
  for episode in xrange(EPISODE):
    Rr = 0.2
    if counter > 300:
      break
    # initialize task
    state = np.array([40.0,50.0,0.0,0.0,0.0,200])
    for step in xrange(STEP):
      counter = counter + 1
      if counter > 300:
        break
      print(" counter is "  + str(counter))
      action = agent.egreedy_action(state) # e-greedy action for train
      if action == 0:
        Rr = 0.20
      elif action == 1:
        Rr = 0.22
      elif action == 2:
        Rr = 0.24
      elif action == 3:
        Rr = 0.26
      elif action == 4:
        Rr = 0.28
      elif action == 5:
        Rr = 0.30
      elif action == 6:
        Rr = 0.32
      elif action == 7:
        Rr = 0.34
      elif action == 8:
        Rr = 0.36
      elif action == 9:
        Rr = 0.4
      elif action == 10:
        Rr = 0.42
      elif action == 11:
        Rr = 0.44
      elif action == 12:
        Rr = 0.46
      elif action == 13:
        Rr = 0.48
      elif action == 14:
        Rr = 0.50
      elif action == 15:
        Rr = 0.52
      elif action == 16:
        Rr = 0.54
      elif action == 17:
        Rr = 0.56
      elif action == 18:
        Rr = 0.58
      elif action == 19:
        Rr = 0.60
      elif action == 20:
        Rr = 0.62
      #else:
        #Rr = Rr + 0.00001
      #if Rr > 0.5:
        #Rr = 0.5
      #elif Rr < 0.1:
        #Rr = 0.1
      eng.workspace['Rr'] = Rr
      #state = [old_vdreal, old_vqreal]
      #next_state,reward,done,_ = env.step(action)
      eng.sim('qxenv.mdl')
      #spd = eng.workspace['speed']
      tor = eng.workspace['torque']
      vd = eng.workspace['vd']
      vq = eng.workspace['vq']
      power = eng.workspace['power']
      vdnp = np.array(vd)
      vdnp_reshape = vdnp.reshape(vdnp.shape[0])
      vdpd = DataFrame(vdnp_reshape)
      vdcut = vdpd.ix[4000:5000]
      vdreal = vdcut.mean()
      old_vdreal = vdreal
      vqnp = np.array(vq)
      vqnp_reshape = vqnp.reshape(vqnp.shape[0])
      vqpd = DataFrame(vqnp_reshape)
      vqcut = vqpd.ix[4000:5000]
      vqreal = vqcut.mean()
      old_vqreal = vqreal
      powernp = np.array(power)
      powernp_reshape = powernp.reshape(powernp.shape[0])
      powerpd = DataFrame(powernp_reshape)
      powercut = powerpd.ix[4000:5000]
      powerreal = powercut.mean()
      old_powerreal = powerreal
      #spdnp = np.array(spd)
      #spdnp_reshape = spdnp.reshape(spdnp.shape[0])
      #spdpd = DataFrame(spdnp_reshape)
      #spdpd.columns = ['speed']
      tornp = np.array(tor)
      tornp_reshape = tornp.reshape(tornp.shape[0])
      torpd = DataFrame(tornp_reshape)
      #torpd.columns = ['torque']
      torcut = torpd.ix[4000:5000]
      torreal = torcut.mean()
      #realobj = pd.concat([obj,old_obj], axis = 1)
      #print(old_obj)
      #realobj.to_csv('aa.csv')
      #old_obj = realobj
      #print(env.step(action))
      # Define reward for agent
      #reward_agent = -1 if done else 0.1
      
      #next_state = np.array([vdreal,vqreal]).reshape(1,2)[0]
      next_state = np.array([40.0,50.0,vdreal,vqreal,powerreal,torreal]).reshape(1,6)[0]
      #state_1 = np.array([vdreal,vqreal])[0]
      #state_2 = next_state.reshape(1,2)[0]
      tor_now = float(torreal)
      
      #reward = (tor_now - tor_max)*10
      #if tor_now > tor_max:
        #tor_max = tor_now
      reward = tor_now - 200
      #reward = tor_now 
      print("tor_now is " + str(tor_now))
      #print("tor_max is " + str(tor_max))
      print("reward is " + str(reward))
      
      agent.perceive(state,action,reward,next_state)
      state = next_state
      #tor_old = tor_now
      print("Rr is " + str(Rr))
      Rr_buffer.append(Rr)
      if reward <-20:
        break
    
  #plt.plot(Rr_buffer)
  #fig = plt.figure
  #ax1 = fig.add_subplot(1,2,1)
  #ax2 = fig.add_subplot(1,2,2)
  #plt.plot(cost_batch)
  #plt.show()
  #ax1.plot(Rr_buffer)
  #ax2.plot(cost_batch)
      #if done:
        #break
    # Test every 100 episodes
  Rr_rec = DataFrame(Rr_buffer, columns = ['Rr'])
  Rr_rec.to_csv('Rr_record.csv')
      
    

if __name__ == '__main__':
  main()

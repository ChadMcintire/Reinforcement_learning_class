
import math  
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = 'C:/Users/nickf/Dropbox/Classes/RL/results/'


class RandomAgent(object):
    def __init__(self, environment):
        self.environment = environment

    def act(self, _):
        return self.environment.action_space.sample()

    def learn(self, state, next_state, action, reward):
        return


class TabularAgent(object):
    def __init__(self, environment):
            # Set up figure
        self.fig = plt.figure(figsize=(40,8), facecolor='white')
        self.qFunction0 = self.fig.add_subplot(1,5,1, frameon=True)
        self.qFunction0.set_title("Q for action 0")
        self.qFunction0.set_xticks([])
        self.qFunction0.set_yticks([])
        self.qFunction1 = self.fig.add_subplot(1,5,2, frameon=True)
        self.qFunction1.set_title("Q for action 1")
        self.qFunction1.set_xticks([])
        self.qFunction1.set_yticks([])
        self.qFunction2 = self.fig.add_subplot(1,5,3, frameon=True)        
        self.qFunction2.set_title("Q for action 2")
        self.qFunction2.set_xticks([])
        self.qFunction2.set_yticks([])
        self.policy = self.fig.add_subplot(1,5,4, frameon=True)   
        self.policy.set_title("Policy b=left, r=right")
        self.policy.set_xticks([])
        self.policy.set_yticks([])
        self.reward = self.fig.add_subplot(1,5,5, frameon=True)  
        self.reward.set_title("Reward vs. learning its")
        plt.show(block=False)
        # setup the hyper parameters
        self.environment = environment
        self.exploreRate = 1.0
        self.learningRate = 0.24
        self.maxBoxes = 20
        # keep track of rewards for visualization
        self.rewardList = []
        self.oneReward = 0
        self.Q = np.zeros(shape=(self.maxBoxes+1, self.maxBoxes+1, self.environment.action_space.n))
        self.steps_done = 0
        self.EPS_END = .001
        self.EPS_START = 1
        self.EPS_DECAY = .005
        #self.Q = np.random.uniform(low = -1, high = 1, size0= (self.maxBoxes+1, self.maxBoxes+1, self.environment.action_space.n))


    def convertState(self, state):
        #     Observation:
        # Type: Box(2)
        # Num    Observation               Min            Max
        # 0      Car Position              -1.2           0.6
        # 1      Car Velocity              -0.07          0.07
        # converts the continuous values to integers for the table
        return (self.convert(state[0], self.environment.observation_space.low[0], self.environment.observation_space.high[0], self.maxBoxes), 
                self.convert(state[1], self.environment.observation_space.low[1], self.environment.observation_space.high[1], self.maxBoxes))
        
    def convert(self, value, minV, maxV, howMany):
        return int(round(howMany*(value - minV)/(maxV - minV)))
        
    def act(self, stateC):
        self.steps_done += 1
        state = self.convertState(stateC)
        threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done * self.EPS_DECAY)

        if np.random.random() < 1 - threshold:
            return np.argmax(self.Q[state[0], state[1]]) 
        else:
            return np.random.randint(0, self.environment.action_space.n)

    def learn(self, stateC, nextStateC, action, reward, done, iteration):
        state = self.convertState(stateC)
        nextState = self.convertState(nextStateC)
        # use the Bellman relationship to get a new estimate for Q
        qEstimate = reward + max(self.Q[nextState[0],nextState[1]])
        # incrementally update the Q value in the table
        self.Q[state[0], state[1], action] += self.learningRate * (qEstimate - self.Q[state[0], state[1], action])
        self.oneReward += reward
        if done:
            # reduce th explore rate
            #print("reward = %d" % self.oneReward)
            #self.exploreRate = max(self.exploreRate - 0.8/1000, 0)
            self.rewardList.append(self.oneReward)
            self.oneReward = 0
            if iteration % 500 == 0:
                print("Updating graphs at %d iterations" % iteration)
                self.qFunction0.imshow(self.Q[:,:,0], cmap='jet', vmin=-210, vmax=0)
                self.qFunction1.imshow(self.Q[:,:,1], cmap='jet', vmin=-210, vmax=0)
                self.qFunction2.imshow(self.Q[:,:,2], cmap='jet', vmin=-210, vmax=0)
                self.policy.imshow(np.argmax(self.Q, axis = 2), cmap='bwr', vmin=0, vmax=2)
                self.reward.plot([i for i in range(0,len(self.rewardList))], self.rewardList, c = 'k')
                self.reward.set_ylim(-210, -100)
                self.saveImageOne(iteration)
                plt.draw()
                plt.pause(0.001)
                
                
    def saveImageOne(self, iteration):
    #print pathOut + fileName
        fileName = "MC_" + str(iteration).rjust(6,'0')
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        #print(onePath + fileName + '.png')
        self.fig.savefig(PATH + fileName + '.png',  dpi=100)
            
        



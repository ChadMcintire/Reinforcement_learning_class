import os, shutil, imageio
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time


class RandomAgent(object):
    def __init__(self, environment):
        self.environment = environment

    #  *_  eats any number of arguments
    def act(self, *_):
        return self.environment.action_space.sample()

    def convert_state(self, state):
        return state

    def learn(self, *_):
        return


class TabularAgent(RandomAgent):
    def __init__(self, environment):
        self.environment = environment
        self.action_space = 3
        self.state_space = 20
        agent_name = "Tabular"

        self.learning_rate = 0.1
        self.discount = 0.95

        self.min_exploration_rate = 0.01
        self.exploration_rate = 1.0
        #self.exploration_decay = 1 - (1e-7 *9)
        self.exploration_decay = 1 - 1e-6

        # Initialize a table to hold an expected value for every state-action pair.
        self.quality_table = np.zeros(
            shape=(self.state_space, self.state_space, self.action_space)
        )

        # Create a temporary folder for storing images
        self.file_names = []
        self.trajectory = []
        self.reward_list = []
        self.average_reward_list = []
        self.total_reward = 0
        self.plotting_iterations = 250
        self.image_path = "./" + agent_name
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

    # Explore early on, then exploit the learned knowledge
    # Actions:
    #   Type: Discrete(3)
    #   Num    Action
    #   0      Accelerate to the Left
    #   1      Don't accelerate
    #   2      Accelerate to the Right
    def act(self, state):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)
        else:
            actions = self.quality_table[tuple(self.state_to_index(state))]
            return np.argmax(actions)

    # Converts the continuous values to integers for the table
    # Observation:
    #   Type: Box(2)
    #   Num    Observation               Min            Max
    #   0      Car Position              -1.2           0.6
    #   1      Car Velocity              -0.07          0.07
    def state_to_index(self, state):
        position, velocity = state.copy()

        new_min, new_max = +0, +self.state_space
        linear_scaling = lambda x, old_min, old_max: np.trunc(
            np.interp(x, (old_min, old_max), (new_min, new_max))
        )
        position = linear_scaling(
            position,
            self.environment.observation_space.low[0],
            self.environment.observation_space.high[0],
        )
        velocity = linear_scaling(
            velocity,
            self.environment.observation_space.low[1],
            self.environment.observation_space.high[1],
        )

        return int(position), int(velocity)

    def learn(self, state, next_state, action, reward, *_):
        self.trajectory.append(self.state_to_index(state))  # plotting purposes
        self.total_reward += reward

    def finish_iteration(self, iteration):
        self.trajectory = np.array(self.trajectory)
        self.reward_list.append(self.total_reward)
        self.average_reward_list.append(np.mean(self.reward_list[:-250]))

        if (iteration + 1) % self.plotting_iterations == 0:
            self.plot(iteration + 1)

        self.trajectory = []
        self.total_reward = 0

    def plot(self, iteration):
        fig = plt.figure(figsize=(20, 4), facecolor="white")
        fig.subplots_adjust(wspace=1)
        fig.suptitle(f"Iteration {iteration}")

        quality_left = fig.add_subplot(1, 5, 1)
        quality_left.imshow(self.quality_table[:, :, 0].T, cmap="Spectral")
        quality_left.set_title("Quality for moving left")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        quality_left = fig.add_subplot(1, 5, 2)
        quality_left.imshow(self.quality_table[:, :, 1].T, cmap="Spectral")
        quality_left.set_title("Quality for not moving")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        quality_left = fig.add_subplot(1, 5, 3)
        quality_left.imshow(self.quality_table[:, :, 2].T, cmap="Spectral")
        quality_left.set_title("Quality for moving right")
        quality_left.set_xticks([])
        quality_left.set_yticks([])
        quality_left.set_xlabel("velocity")
        quality_left.set_ylabel("position")

        policy = fig.add_subplot(1, 5, 4)
        policy.imshow(np.argmax(self.quality_table, axis=2).T, cmap="Spectral")
        policy.plot(
            self.trajectory[:, 0], self.trajectory[:, 1], c="k", linewidth=2,
        )
        policy.set_title("Policy: r=left, w=neutral, b=right")
        policy.set_xticks([])
        policy.set_yticks([])
        policy.set_xlabel("velocity")
        policy.set_ylabel("position")

        reward = fig.add_subplot(1, 5, 5)
        reward.plot(np.arange(len(self.reward_list)), self.reward_list, c="k")
        reward.plot(
            np.arange(len(self.reward_list)),
            self.average_reward_list,
            c="r",
            linewidth=2,
        )
        reward.set_title("Rewards over time")
        reward.set_xlabel("iterations")
        reward.set_ylabel("reward")
        reward.set_ylim([-200, 0])

        file_name = f"{self.image_path}/{iteration}.png"
        self.file_names.append(file_name)
        plt.savefig(file_name)
        plt.close("all")

    def make_animations(self):
        with imageio.get_writer("agent_learning.gif", mode="I") as writer:
            for file_name in self.file_names:
                image = imageio.imread(file_name)
                writer.append_data(image)


# Monte-Carlo doesn't quite work with the Mountain Car problem because
# when we take an action we don't know for certain what the next state is
class TabularAgentMonteCarlo(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)
        agent_name = "MCES"
        self.image_path = "./" +agent_name
        self.rewards = []
        self.states = []
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

        # Monte-Carlo looks at Value estimation instead of Quality estimation
        self.value_table = np.zeros(shape=(self.state_space, self.state_space))

    # Explore early on, then exploit the learned knowledge
    # Actions:
    #   Type: Discrete(3)
    #   Num    Action
    #   0      Accelerate to the Left
    #   1      Don't accelerate
    #   2      Accelerate to the Right
    def act(self, state):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        return np.random.randint(self.action_space)

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        self.rewards.append(reward)
        self.states.append(self.state_to_index(state))

        if done:
            for index in range(len(self.states)):
                rewards = 0
                for t in range(index, len(self.rewards)):
                    rewards += (self.discount ** t) * self.rewards[t]
                self.value_table[self.states[index]] += self.learning_rate * (
                    rewards - self.value_table[self.states[index]]
                )

    def finish_iteration(self, iteration):
        super().finish_iteration(iteration)

        self.rewards = []
        self.states = []

    def plot(self, iteration):
        fig = plt.figure(figsize=(8, 4), facecolor="white")
        fig.subplots_adjust(wspace=1)
        fig.suptitle(f"Iteration {iteration}")

        value = fig.add_subplot(1, 2, 1)
        value.imshow(self.value_table.T, cmap="Spectral")
        value.plot(
            self.trajectory[:, 0], self.trajectory[:, 1], c="k", linewidth=2,
        )
        value.set_title("Value table")
        value.set_xticks([])
        value.set_yticks([])
        value.set_xlabel("velocity")
        value.set_ylabel("position")

        reward = fig.add_subplot(1, 2, 2)
        reward.plot(np.arange(len(self.reward_list)), self.reward_list, c="k")
        reward.plot(
            np.arange(len(self.reward_list)),
            self.average_reward_list,
            c="r",
            linewidth=2,
        )
        reward.set_title("Rewards over time")
        reward.set_xlabel("iterations")
        reward.set_ylabel("reward")
        reward.set_ylim([-200, 0])

        file_name = f"{self.image_path}/{iteration}.png"
        self.file_names.append(file_name)
        plt.savefig(file_name)
        plt.close("all")


# On-policy Temporal Difference is also known as: SARSA
class TabularAgentOnPolicyTD(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)
        agent_name = "Sarsa"
        self.image_path = "./" +agent_name
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        next_action = self.act(next_state)
        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)

        update = self.learning_rate * (
            reward
            + self.discount * self.quality_table[next_state + (next_action,)]
            - self.quality_table[state + (action,)]
        )
        self.quality_table[state + (action,)] += update


# Off-policy Temporal Difference is also known as: MAXSARSA and Q-Learning
class TabularAgentOffPolicyTD(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)
        agent_name = "./Q-learning"
        self.image_path = "./" +agent_name
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)


    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)

        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)

        update = self.learning_rate * (
            reward
            + self.discount * np.max(self.quality_table[next_state])
            - self.quality_table[state + (action,)]
        )
        self.quality_table[state + (action,)] += update

# n-step 
class TabularNstep(TabularAgent):
    
    #loop for each episode
    #initialize and store an action 
    #Set T to infinite
    #loop for the episode for t= 0, 1, 2, ...;
        # if t < T
            #Take action A_t
            #store next reward and next state
            #if the next state is the terminal state
                #set T to that value
            #else
                #select print("learn time --- %s seconds ---" % (time.time() - start_time))and store the action A_t+1
        # tao = t -n +1
        # if tao >= 0
            # G = 
            # G = 
            # Update Q
            # use the greedy policy wrt Q
    # until tao = t-1

    def __init__(self, environment):
        super().__init__(environment)
        agent_name = "Nstep"

        self.T = np.inf
        self.t = 0
        self.n = 3
        self.rewards = []
        self.states = []
        self.actions = []
        self.gamma = .99
        self.learning_rate = .01
        self.image_path = "./" +agent_name
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)


    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)
        
        if not self.states and not self.actions and not self.rewards:
            self.actions.append(action)
            self.rewards.append(reward)
            self.states.append(self.state_to_index(state))

        if self.t < self.T:
            self.rewards.append(reward)
            self.states.append(self.state_to_index(next_state))

            if done:
                self.T = self.t + 1
            else: 
                action = self.act(next_state)  
                self.actions.append(action)
        tau = self.t - self.n + 1
        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + self.n +1, self.T + 1)):
                #print("i", i)
                #print("t", self.t)
                #print(self.rewards)
                G += np.power(self.gamma, i - tau - 1) * self.rewards[i]
            if tau + self.n < self.T:
                state_action = (self.states[tau + self.n], self.actions[tau + self.n])
                G += np.power(self.gamma, self.n) * self.quality_table[state_action[0] + (state_action[1],)]
            # update Q values
            state_action = (self.states[tau], self.actions[tau])
            #print(state_action[0] + (state_action[1],))
            #print("state + action" , state + (action,))
                
            update = self.learning_rate * (
            G
            - self.quality_table[state_action[0] + (state_action[1],)]
            )

            self.quality_table[state_action[0] + (state_action[1],)] += update  

        #if tau == self.T - 1:
        #    break
        
        self.t += 1


    def finish_iteration(self, iteration):
        super().finish_iteration(iteration)

        self.rewards = []
        self.states = []
        self.actions = []
        self.t = 0
        self.T = np.inf



class TabularDynaQ(TabularAgent):
    def __init__(self, environment):
        super().__init__(environment)
        self.n = 20
        self.maxBoxes = 20
        self.discount =.99
        self.learning_rate = .001
        self.nextStateReward = [[[dict(), dict(), dict()] for _ in range(self.maxBoxes)] for _ in range(self.maxBoxes)]
        update_state_action = False
        self.state_action = []
        agent_name = "DynaQ"
        self.image_path = "./" + agent_name
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.mkdir(self.image_path)

    def learn(self, state, next_state, action, reward, done):
        super().learn(state, next_state, action, reward, done)
        #start_time = time.time()

        state = self.state_to_index(state)
        next_state = self.state_to_index(next_state)

        state_action = state + (action,)
        self.Qupdate(state_action, next_state, reward)

        if not next_state in self.nextStateReward[state[0]][state[1]][action]:
            self.nextStateReward[state[0]][state[1]][action][next_state] = reward
            update_state_action = True
        else:
            update_state_action = False
        
        if update_state_action:
            self.state_action = []
            for x in range(np.shape(self.nextStateReward)[0]):
                for xDot in range(np.shape(self.nextStateReward)[1]):
                    for a in range(self.environment.action_space.n):
                        if self.nextStateReward[x][xDot][a]:
                            self.state_action.append((x,xDot,a))

        #print("learn time --- %s seconds ---" % (time.time() - start_time))
        self.planning() 


    def planning(self):
        #start_time = time.time()
        for i in range(self.n):
            state_action_choice = random.choice(self.state_action)
            next_state_key = self.nextStateReward[state_action_choice[0]][state_action_choice[1]][state_action_choice[2]].keys()
            next_state = list(next_state_key)[0]
            reward = self.nextStateReward[state_action_choice[0]][state_action_choice[1]][state_action_choice[2]][next_state]
            #remove the action from state action to get the state
            #state = state_action_choice[:-1]
            
            self.Qupdate(state_action_choice, next_state, reward)
        #print("plan time --- %s seconds ---" % (time.time() - start_time))

    def finish_iteration(self, iteration):
        super().finish_iteration(iteration)

        update_state_action = False 
        print(np.shape(self.state_action))
        #update_state_action = True
        #self.nextStateReward = [[[dict(), dict(), dict()] for _ in range(self.maxBoxes)] for _ in range(self.maxBoxes)]
        #self.state_action = []

    def Qupdate(self, state_action, next_state, reward):
            update = self.learning_rate * (
            reward
            + self.discount * np.max(self.quality_table[next_state])
            - self.quality_table[state_action]
            )
            self.quality_table[state_action] += update


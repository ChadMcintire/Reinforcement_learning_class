# Local imports
from agents import RandomAgent
from agents import TabularAgent

# Python imports
import numpy as np
from time import time  # just to have timestamps in the files
import matplotlib.pyplot as plt

# OpenAI Gym imports
import gym
from gym import wrappers

# Remove the monitoring if you do not want a video
env = gym.make("MountainCar-v0")
env = wrappers.Monitor(env, "./videos/" + str(time()) + "/")


# Change the agent to a different one by simply swapping out the class
# ex) RandomAgent(env) --> TabularAgent(env)
#agent = RandomAgent(env)
agent = TabularAgent(env)

# We are only doing a single simulation. Increase 1 -> N to get more runs.


def plot_durations(x_scatter, y_scatter):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Average Duration Over 200 episodes')
    plt.plot(x_scatter, y_scatter)
    plt.savefig("plot of durs")



from collections import deque

moving_200 = deque()
moving_200_complete = deque()
total_episodic_completion = 0
episodes200 = []
completions_of_200 =[]
for iteration in range(10001):
    #print(iteration)
    # Always start the simulation by resetting it
    state = env.reset()
    done = False
    cummulative_reward = []

    # Either limit the number of simulation steps via a "for" loop, or continue
    # the simulation until a failure state is reached with a "while" loop
    while not done:

        # Render the environment. You will want to remove this or limit it to the
        # last simulation iteration with a "if iteration == last_one: env.render()"
        if iteration % 500 == 0:
            env.render()

        # Have the agent
        #   1: determine how to act on the current state
        #   2: go to the next state with that action
        #   3: learn from feedback received if that was a good action
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        if next_state[0] >= 0.4:
            reward += .5

        if next_state[0] >= 0.5:
            reward += 2
            moving_200_complete.append(1)
    

        cummulative_reward.append(reward)
        agent.learn(state, next_state, action, reward, done, iteration)

        # Progress to the next state
        state = next_state

    moving_200.append(np.sum(cummulative_reward))

    if iteration % 200 == 0:
        print()
        print("episode: ", iteration)
        print("moving 200: ", np.mean(moving_200))
        moving_200.clear()

        print("moving 200 comp: ", np.sum(moving_200_complete))
        completions_of_200.append(np.sum(moving_200_complete))
        episodes200.append(iteration)
        print(completions_of_200)
        print(episodes200)
        plot_durations(episodes200, completions_of_200)
        total_episodic_completion += np.sum(moving_200_complete)
        moving_200_complete.clear()


    #env.render()  # Render the last simluation frame.
env.close()  # Do not forget this! Tutorials like to leave it out.
print(total_episodic_completion)

"""
MountainCar Movements
    0 - Move left
    1 - Don't move
    2 - Move right
"""
#print(agent.value_table)

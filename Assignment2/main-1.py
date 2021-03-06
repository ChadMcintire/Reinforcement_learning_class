# Local imports
from agent import *

# OpenAI Gym imports
import gym
from gym import wrappers

# Remove the monitoring if you do not want a video
environment = gym.make("MountainCar-v0")
environment = wrappers.Monitor(
    environment,  # environment to watch
    f"./videos/",  # where to save videos
    force=True,  # clear old videos
    video_callable=lambda episode_id: episode_id == 29_999,  # what run to record
)
agent = TabularAgentOnPolicyTD(environment)

for iteration in range(30_000):
    print(f"Iteration: {iteration}, Exploration Rate: {agent.exploration_rate:.7f}")

    state = environment.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.learn(state, next_state, action, reward, done)
        state = next_state

    agent.finish_iteration(iteration)

agent.make_animations()
environment.close()

from ddpg_torch import Agent, transform_obs
from utils import plotLearning
from fileinput import filename
# from reg_env import Env
import matplotlib.pyplot as plt
from reg_env import preprocess_data
import json
import random
import numpy as np
import torch
import gc

def save_episode_data_json(episode, durations, scores):
    data = {"episode": episode, "durations": durations, "scores": scores}
    print("Saving episode data to json file")
    filename = "tmp/episode_data_dict.json"
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_episode_data_json():
    filename = "tmp/episode_data_dict.json"
    print("Loading episode data from json file")
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


EPSILON = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.15

torch.cuda.empty_cache()

envs = preprocess_data('coffee', display=False)
env = envs[0]
agent = Agent(alpha=0.000025, beta=0.00025, max_size=100000, input_dims=[3,112,112], tau=0.001,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

agent.load_models()
np.random.seed(0)


score_history = []
episode_history = []
episode_data = load_episode_data_json()
episode = episode_data["episode"] +1
score_history = episode_data["scores"]
episode_history = episode_data["durations"]

# print the device
print(f"device: {agent.actor.device}")

for i in range(episode, 1000):
    EPSILON = 0.95
    obs = env.reset()
    done = False
    score = 0
    env = random.choice(envs)
    t = 0
    while not done:
        t += 1
        # choose action by epsilon greedy
        if np.random.random() > EPSILON:
            act = agent.choose_action(obs)
        else:
            act = np.array([np.array([random.uniform(-1,1), random.uniform(-1,1)])])

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        # act = agent.choose_action(obs)
        new_state, reward, done = env.step(act)
        agent.remember(transform_obs(obs), act, reward, transform_obs(new_state), int(done))
        agent.learn()
        score += reward
        obs = new_state
        if t%100 == 0:
            x_amount = act[0][0]*env.fixed_image.size[0]
            y_amount = act[0][1]*env.fixed_image.size[1]
            print(f'Episode: {i}.{t} Score: {score.round(2)}, reward: {reward} x: {x_amount.round(2)}, y: {y_amount.round(2)} epsilon: {EPSILON}')
            # plot the state
        if t%10000 == 0:
            # plot the state
            env = random.choice(envs)
            EPSILON = 0.95
            # plt.imshow(obs)
            # plt.show()
            
        #env.render()
    score_history.append(score)
    episode_history.append(t)

    # plot and save the score
    plt.plot(episode_history)
    plt.savefig('runtime_episodes.png')
    
    if i % 2 == 0:
       agent.save_models()
       save_episode_data_json(i, episode_history, score_history)

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
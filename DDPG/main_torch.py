from ddpg_torch import Agent, transform_obs
from utils import plotLearning
from fileinput import filename
# from reg_env import Env
import matplotlib.pyplot as plt
from reg_env import preprocess_data, printProgressBar
import json
import random
import numpy as np
import torch
import time
import gc

def save_episode_data_json(episode, durations):
    data = {"episode": episode, "durations": durations}
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
EPSILON_DECAY = 0.9975
EPSILON_MIN = 0.15

torch.cuda.empty_cache()

envs = preprocess_data('1000_random', display=False)
env = envs[0]
agent = Agent(alpha=0.000025, beta=0.00025, max_size=100000, input_dims=[3,112,112], tau=0.001,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

# agent.load_models()
np.random.seed(0)

# load test history if exists
filename = "tmp/test_history.json"
try:
    with open(filename, 'r') as f:
        test_history = json.load(f)
except:
    test_history = []

score_history = []
episode_history = []
episode = 0


# print the device
print(f"device: {agent.actor.device}")

def train(load_models = True):
    episode_data = load_episode_data_json()
    episode = episode_data["episode"] +1
    episode_history = episode_data["durations"]
    if load_models:
        agent.load_models()
    for i in range(episode, 10000):
        EPSILON = 0.95
        env = random.choice(envs)
        obs = env.reset()
        # plt the start pic
        done = False
        score = 0
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
            if t%1000 == 0:
                done = True
                
            if t%100 == 0:
                reset_message = ""
                if done:
                    reset_message = "Env Reset "
                x_amount = act[0][0]*env.fixed_image.size[0]
                y_amount = act[0][1]*env.fixed_image.size[1]
                # get current hour and minute
                current_time = time.localtime()
                
                print(f'{reset_message}Episode: {i}.{t} , reward: {reward} x: {x_amount.round(2)}, y: {y_amount.round(2)} epsilon: {EPSILON}, time: {time.localtime().tm_hour}:{time.localtime().tm_min}')
                # plot the state
            
            # if t%10000 == 0:
            #     # plot the state
            #     done = True
            #     EPSILON = 0.95
            #     # plt.imshow(obs)
            #     # plt.show()
                
            #env.render()
        inference(episode=i, iters=1)
        # score_history.append(score)
        episode_history.append(t)

        # plot and save the score
        plt.plot(episode_history)
        plt.savefig('runtime_episodes.png')
        save_episode_data_json(i, episode_history)

        if i % 1 == 0:
            agent.save_models()
            # save agent memory
            # agent.save_memory()
            
            
            
        # clear memory once in 5 episodes
        if i%5 == 0:
            print("Clearing memory")
            gc.collect()
            torch.cuda.empty_cache()
            # load models
            agent.load_models()
            episode_data = load_episode_data_json()
            episode = episode_data["episode"] +1
            episode_history = episode_data["durations"]
            # agent.load_memory()



def inference(episode=0 ,iters=100):
    # load the model
    # agent.load_models()
    # inference
    for i in range(iters):
        env = random.choice(envs)
        obs = env.reset()
        env_start_pic = obs
        done = False
        score = 0
        t = 0
        rewards = []
        while not done:
            
            t += 1
            t_copy = t
            act = agent.choose_action(obs, add_noise=True)
            new_state, reward, done = env.step(act)
            score += reward
            if done:
                t_copy = 10000
                reward = 0 
                rewards.append(reward)
            if t%2 == 0:
                rewards.append(reward)
            
            obs = new_state
            if t%10000 == 0:
                done = True
                
            # if t%50 == 0:
            printProgressBar(t_copy, 10000, prefix='Inference:', suffix=f' {t} steps', length=100)
            #     print(f'Episode: {i}.{t} Score: {score.round(2)}, reward: {reward}')
        print(f'#### Test Episode: {episode}, Steps: {t} ####')
        test_history.append(t)
        # plot the history 
        plt.plot(test_history)
        plt.title("Inference history")
        plt.savefig(f"tmp/runtime_inference.png")

        # save test history
        filename = "tmp/test_history.json"
        with open(filename, 'w') as f:
            json.dump(test_history, f)
        
        # plt.plot(rewards)
        # plt.title("rewards")
        # plt.savefig(f"tmp/rewards/rewards_{i}.png")
        # reset plot
        plt.clf()
        # plt.imshow(env_start_pic)
        # plt.title("start environment")
        # plt.savefig(f"tmp/start_pics/start_pic_{i}.png")
        # plt.clf()
        # combine the two plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Vertically stacked subplots')
        ax1.plot(rewards)
        ax1.set_title("rewards")
        ax2.imshow(env_start_pic)
        ax2.set_title("start environment")
        plt.savefig(f"tmp/combined/combined_{i}.png")
        plt.clf()
        """


def main():
    train()
    # inference()

if __name__ == '__main__':
    main()
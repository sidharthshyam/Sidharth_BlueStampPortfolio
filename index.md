# Lunar Landar Reinforcement Learning
Replace this text with a brief description (2-3 sentences) of your project. This description should draw the reader in and make them interested in what you've built. You can include what the biggest challenges, takeaways, and triumphs from completing the project were. As you complete your portfolio, remember your audience is less familiar than you are with all that your project entails!

You should comment out all portions of your portfolio that you have not completed yet, as well as any instructions:
```HTML 
<!--- This is an HTML comment in Markdown -->
<!--- Anything between these symbols will not render on the published site -->
```

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Sidharth S | Skyline High School | Computer Science and AI/ML | Rising Sophomore |

**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](logo.svg)
  
# Final Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/ud03IRiQ_f0?si=0etjGS8OP72odx-D" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE



# Second Milestone


<iframe width="560" height="315" src="https://www.youtube.com/embed/ODmsfNxn-ZA?si=vFSmYqU-ojKt6ScV" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

During this milestone, I focused on optimizing the DQN model to achieve the best possible performance. Initially, I attempted this manually—adjusting hyperparameters like learning rate and exploration decay, then retraining the model with each change. However, I quickly realized this approach was time-consuming and inefficient.

To streamline the process, I developed an automated Python script (hyperparameter_tuning.py) that automatically loops through combinations of key hyperparameters: learning rate, exploration fraction, and neural network architecture. Each configuration was first trained for 20,000 steps, allowing me to identify and eliminate weak performers early on. I then reran the most promising configurations for 100,000 steps to fully evaluate their potential.

After each run, the script parsed the monitor.csv log, calculated the mean episode reward and length, and saved the results of each configuration to a master CSV file. To visualize performance trends, the code also generates a heatmap of mean reward across learning rate and exploration fraction values, which provided a clear, visual view of which hyperparameter regions yielded the best learning outcomes.

# Tables and Diagrams
![Milestone 2 Diagram](Lunar%20Lander%20Mean%20Reward%20Heatmap.png)


# Milestone 2 Code
Here's the code for the hyperparameter_tuning.py file.
```
#!/usr/bin/env python3
import os
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 1) Hyperparameter grid
learning_rates      = [0.001, 0.005, 0.0005]
exploration_fracs   = [0.5, 0.3, 0.7]
nn_layer_configs    = [ [128,128], [64,64,64], [64,64]]

# 2) Paths
LOG_DIR    = "./tmp_gym/"
os.makedirs(LOG_DIR, exist_ok=True)
RESULT_CSV = "hyperparam_tuning_results.csv"

# 3) Init results CSV
pd.DataFrame(
    columns=[
        "learning_rate","exploration_fraction","nn_layers",
        "mean_reward","mean_ep_length","timesteps"
    ]
).to_csv(RESULT_CSV, index=False)

# 4) Loop over each combo
for lr, ef, layers in itertools.product(learning_rates, exploration_fracs, nn_layer_configs):
    # a) create fresh env + monitor
    env = gym.make("LunarLander-v3")
    env = Monitor(env, LOG_DIR)
    
    # b) evaluation callback (optional saving best model)
    callback = EvalCallback(
        env,
        eval_freq=10000,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        deterministic=True,
        verbose=0
    )
    
    # c) build & train
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=layers)
    model = DQN(
        "MlpPolicy", env,
        learning_rate=lr,
        exploration_initial_eps=1.0,
        exploration_fraction=ef,
        buffer_size=50000,
        batch_size=64,
        train_freq=(4, "step"),
        gradient_steps=4,
        target_update_interval=500,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    TIMESTEPS = 100000
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    
    # d) read the actual monitor.csv (skip any comments)
    monitor_path = os.path.join(LOG_DIR, "monitor.csv")
    df = pd.read_csv(monitor_path, comment='#')
    mean_r   = df["r"].mean()
    mean_len = df["l"].mean()
    
    # e) append to results CSV
    pd.DataFrame([{
        "learning_rate":      lr,
        "exploration_fraction": ef,
        "nn_layers":          str(layers),
        "mean_reward":        mean_r,
        "mean_ep_length":     mean_len,
        "timesteps":          TIMESTEPS
    }]).to_csv(RESULT_CSV, mode="a", header=False, index=False)
    
    # f) clean up before next run
    for f in os.listdir(LOG_DIR):
        os.remove(os.path.join(LOG_DIR, f))

# 5) Plot heatmap of mean rewards
pivot = df_res.pivot_table(
    index="learning_rate",
    columns="exploration_fraction",
    values="mean_reward"
)

plt.figure(figsize=(6,5))
plt.title("Mean Reward Heatmap")
plt.xlabel("Exploration Fraction")
plt.ylabel("Learning Rate")
plt.imshow(pivot, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="Mean Reward")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.tight_layout()
plt.savefig("hyperparam_reward_heatmap.png")
print("▶ Hyperparameter tuning complete.")
print(f"  • Results: {RESULT_CSV}")
print("  • Heatmap: hyperparam_reward_heatmap.png")
```

# First Milestone


<iframe width="560" height="315" src="https://www.youtube.com/embed/ud03IRiQ_f0?si=0etjGS8OP72odx-D" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


I achieved my first milestone by setting up remote access from my computer to the Raspberry Pi using Raspberry Pi Imager and a USB card reader, allowing me to SSH into the Pi and begin development. After establishing the SSH connection, I created a virtual environment to manage dependencies cleanly, and organized the necessary folders and Python files for the project. A key part of this milestone was gaining a solid understanding of the codebase—specifically, the purpose of each import, how the LunarLander-v3 game environment is initialized, and how the neural network and training loop are structured using the DQN algorithm. Once I grasped these core components, I ensured that the video recording functionality for the lunar lander simulation and the training progress graph both worked correctly. Although the initial model was not fully optimized and the lander frequently crashed, this step confirmed that the full training pipeline was functional and ready for further improvement.

# Schematics 
This diagram shows the structure of my Lunar Lander reinforcement learning setup using Raspberry Pi and VS Code:

![Milestone 1 Diagram](Initial%20Lunar%20Lander%20FlowChart.drawio.png)



# Milestone 1 Code
Here's the code for the intial Lunar Lander without too much optimization. 
```
# Imports
import io
import os
import glob
import torch
import base64
import minigrid
import IPython.display

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

import sys
import gymnasium

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import *
from gymnasium.wrappers import RecordVideo
from ale_py import ALEInterface

import warnings
warnings.filterwarnings('ignore')

ale = ALEInterface()



# @title Play Video function
from IPython.display import HTML
from base64 import b64encode
from pyvirtualdisplay import Display

# create the directory to store the video(s)
os.makedirs("./video", exist_ok=True)

display = Display(visible=False, size=(1400, 900))
_ = display.start()

"""
Utility functions to enable video recording of gym environment
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""
def render_mp4(videopath: str) -> str:
  """
  Gets a string containing a b4-encoded version of the MP4 video
  at the specified path.
  """
  mp4 = open(videopath, 'rb').read()
  base64_encoded_mp4 = b64encode(mp4).decode()
  return f'<video width=400 controls><source src="data:video/mp4;' \
         f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'

nn_layers = [64, 64, 64]  # This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
                      # If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

learning_rate = 0.0005  # This is the step-size with which the gradient descent is carried out.
                       # Tip: Use smaller step-sizes for larger networks.


log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create environment
env_name = 'LunarLander-v3'
env = gym.make(env_name)
# You can also load other environments like cartpole, MountainCar, Acrobot.
# Refer to https://gym.openai.com/docs/ for descriptions.

# For example, if you would like to load Cartpole,
# just replace the above statement with "env = gym.make('CartPole-v1')".
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

env = stable_baselines3.common.monitor.Monitor(env, log_dir )

callback = EvalCallback(env, log_path=log_dir, deterministic=True)  # For evaluating the performance of the agent periodically and logging the results.
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
            learning_rate=learning_rate,
            batch_size=64,  # for simplicity, we are not doing batch update.
            buffer_size=50000,  # size of experience of replay buffer. Set to 1 as batch update is not done
            learning_starts=10000,  # learning starts immediately!
            gamma=0.99,  # discount facto. range is between 0 and 1.
            tau = 0.005,  # the soft update coefficient for updating the target network
            target_update_interval=750,  # update the target network immediately.
            train_freq=(2,"step"),  # train the network at every step.
            max_grad_norm = 10,  # the maximum value for the gradient clipping
            exploration_initial_eps = 1,  # initial value of random action probability
            exploration_fraction = 0.6,  # fraction of entire training period over which the exploration rate is reduced
            gradient_steps = 10,  # number of gradient steps
            seed = 1,  # seed for the pseudo random generators
            verbose=1)  # Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.

# You can also experiment with other RL algorithms like A2C, PPO, DDPG etc.
# Refer to  https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
# for documentation. For example, if you would like to run DDPG, just replace "DQN" above with "DDPG".


#training the DQN Model
model.learn(total_timesteps=200000, log_interval=10, callback=callback)
# The performance of the training will be printed every 10000 episodes. Change it to 1, if you wish to
# view the performance at every training episode.

env = gym.make(env_name, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="video",
    name_prefix=f"{env_name}_learned",
    episode_trigger=lambda episode_id: True
)

observation, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"\nTotal reward: {total_reward}")

# show video
html = render_mp4(f"video/{env_name}_learned-episode-0.mp4")
HTML(html)

x, y = ts2xy(load_results(log_dir), 'timesteps')  # Organising the logged results in to a clean format for plotting.
plt.plot(x, y)
plt.ylim([-300, 300])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
plt.savefig('training_plot.png')  # Save the plot
print("Plot saved as training_plot.png. Check the file to verify.")
plt.close()  # Clean up

}
```

# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Raspberry Pi kit | Acts as the main computer that is running the project and runs the Python and Reinforcement Learning Framework | $96.99 | <a href="https://www.amazon.com/RasTech-Raspberry-Starter-Heatsink-Screwdriver/dp/B0C8LV6VNZ/ref=sr_1_4?crid=3506HY00MCGVM&dib=eyJ2IjoiMSJ9._zkM62vSQ8p7tNr88715LdMv_qHh72Je-tkF9PXEa3chDE53QT4aZu4AGAb4ihE61QY4ZD55nKF6Fp2Kfs8t7AbafM_JrlJFfHo9OB4eAVGqa0EB-7aoBQHPmhKHZ2MW8ny-Kd44bMVlVxPlTWVk5YHIN5P3uKVqrE5Dcal0rKkHny-O6Xyb5ux2AOU6OwVbkag_bqBX66RQNRrgBuz-0pS43mcx93IZTQA9R8NaJJypYU2HAycp-XicTFmyU60a01Nfm9iuyo6B9yA8ppN3OQQyJ-NQ9xyNPxfTLwkqtng.yAYpU6outhQcZmOZhN9Wb6yTw7A85CNUbXZguGInZNg&dib_tag=se&keywords=raspberry%2Bpi%2Bkit&qid=1718848547&s=electronics&sprefix=rasbperry%2Bpi%2Bkit%2Celectronics%2C83&sr=1-4&th=1"> Link </a> |
| Monitor/Display| It is a portable screen to interact with the Raspberry Pi directly where you can see the real time rendering of the Lunar Lander simulation | $46.99 | <a href="https://www.amazon.com/Hosyond-Display-1024×600-Capacitive-Raspberry/dp/B09XKC53NH/ref=sr_1_3?crid=1KKB9WC62OIAD&keywords=raspberry%2Bpi%2Bips&qid=1685911698&s=electronics&sprefix=raspberry%2Bpi%2Bips%2B%2Celectronics%2C87&sr=1-3&th=1#customerReviews"> Link </a> |
| Mouse/keyboard | These are input devices that connect to the Raspberry Pi to type commands and navigate the Raspberry Pi interface | $21.99 | <a href="https://www.amazon.com/gp/product/B07XDWCLYF/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1"> Link </a> |
| SD Card Adapter USB | Used during the initial setup using the Raspberry Pi Imager to prepare the OS and SSH access | $9.99 | <a href="https://www.amazon.com/dp/B081VHSB2V?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1"> Link </a> |

# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.

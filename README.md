# Reinforcement Learning Workspace

The basic workspace for reinforcement learning with CoppeliaSim (VREP) simulation environments, including some demonstrated project for beginners.
Some tutorials can be found at:

I add gymnasium and ZeroMQ support compared with the original repository. All the demos have been tested on Windows 11, with python 3.10 environment.


---
## Environments setup

You have to install the following softwares and environments for this project, the recommend operating system is Ubuntu.
- CoppeliaSim 4.6 (https://www.coppeliarobotics.com/)
- Python 3.6+ (Anaconda is recommended)
- Gymnasium (https://github.com/Farama-Foundation/Gymnasium)
- Stable-baselines3 (https://github.com/DLR-RM/stable-baselines3)
- Pytorch (https://pytorch.org/)
- Visdom (pip install visdom)


---
## Demo 1: Cart-pole control with the A2C (modified SAC) algorithm

- Step 1: Replace pythonWrapperV2.lua in Coppeliasim path
- Step 2: run CoppeliaSim (using start_copp.bat to start multiple envs)
- Step 3: run visdom in your terminal, open your browser, and visit link: localhost:8097
- Step 4: run the script named 'demo_qbot_learning_multi_SAC_Jump.py' in the sub-path ./examples

Then we have:

![](pic/example.gif)


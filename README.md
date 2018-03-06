# Asynchronous Advantage Actor-Critic Algorithm (A3C)
- Still in progress.

- TODO: Support more network architecture

## Install dependency
- [pytorch 0.3.1](http://pytorch.org/)
- openai-gym

First, type the following command
`pip install -r requirements.txt`

Second, install gym by typing:
`pip install gym`

Third, go to pytorch.org to install pytorch

## execution
`python main.py`

## NOTICE
1. training speed issue, please add `os.environ['OMP_NUM_THREADS'] = '1'` to your source code, if your platform is Linux. Without adding this code, it will take much time for training because of process blocking issue. You cau see related discussion here: https://github.com/ikostrikov/pytorch-a3c/issues/33


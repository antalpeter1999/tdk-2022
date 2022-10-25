# Quadcopter simulation framework based on MuJoCo

## Main use cases:
- Trajectory tracking with geometric control for evaluating results of trajectory planning and swarm applications
- Performing complex maneuvers, such as a backflip
- Reinforcement learning of quadcopter agile maneuvering
- Experiments with payload transportation

## Installation
To install this repository, first open a terminal and run the following commands:
```
$ git clone https://github.com/antalpeter1999/tdk-2022
$ cd tdk-2022/
```
Now create a virtual environment, activate it, then execute:
```
$ pip install -e .
```
To test the installation run a script, e.g.
```
$ cd flight/
$ python fly.py
```
To run the payload transportation script, some modifications are needed in the cvxopt Python package.
# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

'BehavioralClone.py' implement BC, training a three-layer neural network using the data from expert policy, and save it in a file


'runPolicy.py'execute the trained neural network model in the environment


'run_expert.py' also has been changed to include DAgger algorithm

WIP:
DAgger results should be improve with iterations




The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

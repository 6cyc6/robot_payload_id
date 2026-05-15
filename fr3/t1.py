import numpy as np

path = "/home/galois/Downloads/trajectory_2/target_joint_positions.npy"

data = np.load(path)
print(data.shape)
print(data)

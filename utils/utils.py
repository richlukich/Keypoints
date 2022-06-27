import numpy as np
root_dir='/content/'
annotDir = root_dir + 'data'
img_dir = '/content/mpii_human_pose_v1/images'
output_size=64
input_size=256
nJoints=16
accIdxs=np.arange (16)
weight_dir='/content/pyrenet_weights42.h5'
epochs=100
mode="dark"
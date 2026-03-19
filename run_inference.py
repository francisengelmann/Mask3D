import torch
from mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh 

model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load input data
pointcloud_file = 'data/scaniverse.ply'
mesh = load_mesh(pointcloud_file)

# prepare data
data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

# run model
with torch.no_grad():
    outputs = model(data, raw_coordinates=features)
    
# map output to point cloud
labels = map_output_to_pointcloud(mesh, outputs, inverse_map, confidence_threshold=0.6)

import matplotlib.pyplot as plt
import numpy as np

unique, counts = np.unique(labels, return_counts=True)

# save colorized mesh
save_colorized_mesh(mesh, labels, 'data/pcl_labelled2.ply', colormap='scannet200')
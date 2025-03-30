import pickle
import numpy as np

# Set numpy print options to show all elements
np.set_printoptions(threshold=np.inf)

# open model_mano_faces.pkl
model_mano_faces = pickle.load(open('results/model_mano_faces.pkl', 'rb'))

# save as .npy
np.save('config/model_mano_faces.npy', model_mano_faces)

# # open model_cfg.pkl
# model_cfg = pickle.load(open('results/model_cfg.pkl', 'rb'))

# # open model_mano_faces.pkl
# model_mano_faces = pickle.load(open('results/model_mano_faces.pkl', 'rb'))


import numpy as np
import os

# Path to one of your saved landmark files
file_path = os.path.join('data', 'A', 'landmark_0.npy')  # Change index as needed

# Load the .npy file
data = np.load(file_path)

# Show the shape and values
print("Shape:", data.shape)       # Expected: (63,)
print("Values:\n", data)

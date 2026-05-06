import h5py

# CHANGE THIS TO YOUR FILE PATH ON YOUR MACHINE
file_path = "C:/Users/yashj/projects/IXN-Project-Team-Extensify/aloha_dataset/episode_05.hdf5"

print(f"Inspecting: {file_path}")

try:
    # Open the file in read-only mode to prevent accidental overwrites
    with h5py.File(file_path, 'r') as f:
        
        # Callback function applied to every item in the file
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: /{name} | Shape: {obj.shape} | Type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: /{name}")

        # Recursively walk through the HDF5 dictionary tree
        f.visititems(print_structure)
        
except Exception as e:
    print(f"Error opening file: {e}")
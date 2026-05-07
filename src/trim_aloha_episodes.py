import h5py
import os
import glob

# --- CONFIGURATION ---
INPUT_DIR = "aloha_cleaned_dataset"
OUTPUT_DIR = "processed_aloha_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

END_FRAMES = {
    "episode_00.hdf5": 420,
    "episode_01.hdf5": 402,
    "episode_02.hdf5": 430,
    "episode_03.hdf5": 317,
    "episode_04.hdf5": 298,
    "episode_05.hdf5": 412,
    "episode_06.hdf5": 365,
    "episode_07.hdf5": 389,
    "episode_08.hdf5": 334,
    "episode_09.hdf5": 390,
    "episode_10.hdf5": 338,
    "episode_11.hdf5": 278,
    "episode_12.hdf5": 343,
    "episode_13.hdf5": 345,
    "episode_14.hdf5": 266,
    "episode_15.hdf5": 406,
    "episode_16.hdf5": 390,
    "episode_17.hdf5": 364,
    "episode_18.hdf5": 500,
    "episode_19.hdf5": 465,
    "episode_20.hdf5": 440,
    "episode_21.hdf5": 430,
    "episode_22.hdf5": 319,
    "episode_23.hdf5": 543,
    "episode_24.hdf5": 279,
    "episode_25.hdf5": 336,
    "episode_26.hdf5": 449,
    "episode_27.hdf5": 343,
    "episode_28.hdf5": 480,
    "episode_29.hdf5": 318,
    "episode_30.hdf5": 329,
    "episode_31.hdf5": 372,
    "episode_32.hdf5": 494,
    "episode_33.hdf5": 599,
    "episode_34.hdf5": 534,
    "episode_35.hdf5": 300,
    "episode_36.hdf5": 544,
    "episode_37.hdf5": 370,
    "episode_38.hdf5": 456,
    "episode_39.hdf5": 396, 
    "episode_40.hdf5": 347,
    "episode_41.hdf5": 364,
    "episode_42.hdf5": 414,
    "episode_43.hdf5": 235,
    "episode_44.hdf5": 300,
    "episode_45.hdf5": 307,
    "episode_46.hdf5": 346,
    "episode_47.hdf5": 305,
    "episode_48.hdf5": 482,
    "episode_49.hdf5": 388, 
    "episode_50.hdf5": 322, 
    "episode_51.hdf5": 423, 
    "episode_52.hdf5": 372, 
    "episode_53.hdf5": 275, 
    "episode_54.hdf5": 397, 
    "episode_55.hdf5": 218, 
    "episode_56.hdf5": 381, 
    "episode_57.hdf5": 299, 
}

def process_aloha_episode(filename, end_idx):
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    # Start frame is universally 0
    start_idx = 0 
    
    with h5py.File(in_path, 'r') as f_in, h5py.File(out_path, 'w') as f_out:
        
        # 1. COPY HEADERS
        for key in f_in.keys():
            if "headers" in key or key == "camera_names" or key == "modality_index":
                f_in.copy(key, f_out)

        # 2. TRIM ROBOT STATES & ACTIONS
        for key in ['action', 'observations/qpos', 'observations/qvel', 'observations/effort', 'observations/force_torque', 'observations/contact_force']:
            if key in f_in:
                data = f_in[key][()]
                f_out.create_dataset(key, data=data[start_idx:end_idx])

        # 3. TRIM COMPRESSED IMAGES
        cam_names = [name.decode('utf-8') for name in f_in['camera_names'][()]]
        out_cam_group = f_out.create_group('observations/images')
        
        old_compress_len = f_in['compress_len'][()]
        new_compress_len = old_compress_len[:, start_idx:end_idx]
        f_out.create_dataset('compress_len', data=new_compress_len)

        for cam in cam_names:
            raw_cam_data = f_in[f'observations/images/{cam}'][()]
            trimmed_cam_data = raw_cam_data[start_idx:end_idx, :]
            out_cam_group.create_dataset(cam, data=trimmed_cam_data)

    print(f"✅ Fast-Trimmed {filename} | New length: {end_idx} frames.")

# --- EXECUTE ---
for filepath in glob.glob(f"{INPUT_DIR}/*.hdf5"):
    filename = os.path.basename(filepath)
    if filename in END_FRAMES:
        process_aloha_episode(filename, END_FRAMES[filename])
    else:
        print(f"⚠️ Skipped {filename}: End frame not defined in script.")
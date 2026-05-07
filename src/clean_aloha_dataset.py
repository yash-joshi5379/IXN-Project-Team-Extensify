import os
import re
import glob
import shutil

# --- 1. CONFIGURATION ---
SOURCE_DIR = "aloha_dataset"
TARGET_DIR = "aloha_cleaned_dataset"

# Put the episode numbers you want to REMOVE in this list (e.g., 5, 12, 68)
# You don't need to sort them; the script handles it!
BAD_EPISODES = [
    # Example: 4, 17, 22
    0, 3, 8, 9, 13, 21, 31, 32, 33, 39, 50, 60
]

# Create the new clean directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

def filter_and_transfer():
    # Grab all files and sort them to ensure we process in the correct order
    files = glob.glob(os.path.join(SOURCE_DIR, "*.hdf5"))
    files.sort()
    
    if not files:
        print(f"❌ No .hdf5 files found in '{SOURCE_DIR}'.")
        return

    print(f"---  Found {len(files)} total episodes. Filtering out {len(BAD_EPISODES)} bad ones... ---")

    pattern = re.compile(r"episode_(\d+)")
    
    new_sequence_counter = 0
    skipped_count = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        
        if match:
            # Get the original episode number
            original_num = int(match.group(1))
            
            # Check if this episode is in your "dodgy" list
            if original_num in BAD_EPISODES:
                print(f"🗑️  Skipped bad episode: {filename}")
                skipped_count += 1
                continue # Skip the rest of the loop and move to the next file
            
            # If it's a good episode, generate its new sequential name
            new_filename = f"episode_{str(new_sequence_counter).zfill(2)}.hdf5"
            new_file_path = os.path.join(TARGET_DIR, new_filename)
            
            # Copy the file to the new folder
            shutil.copy2(file_path, new_file_path)
            print(f"✅ Copied: {filename} -> {new_filename}")
            
            new_sequence_counter += 1
            
    print("\n--- 🎉 CLEANUP COMPLETE ---")
    print(f"Total original files: {len(files)}")
    print(f"Bad files removed:    {skipped_count}")
    print(f"Clean files saved:    {new_sequence_counter} (Sequenced from episode_00 to episode_{str(new_sequence_counter - 1).zfill(2)})")
    print(f"Location:             {TARGET_DIR}/")

# --- EXECUTE ---
if __name__ == "__main__":
    filter_and_transfer()
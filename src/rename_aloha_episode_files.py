import os
import glob

# --- CONFIGURATION ---
DATA_DIR = "aloha_dataset"

def force_sequential_rename():
    # Find all .hdf5 files
    files = glob.glob(os.path.join(DATA_DIR, "*.hdf5"))
    
    if not files:
        print(f"❌ No .hdf5 files found in '{DATA_DIR}'.")
        return

    # Sort files alphabetically. 
    # The "1_" files will be first, followed by the "2026..." files in chronological order.
    files.sort()

    print(f"--- 🛠️ Found {len(files)} files. Forcing strict sequential rename... ---")

    # STEP 1: Rename everything to a safe temporary name to avoid ANY collisions
    temp_files = []
    for i, file_path in enumerate(files):
        temp_name = os.path.join(DATA_DIR, f"TEMP_RENAME_{i}.hdf5")
        os.rename(file_path, temp_name)
        temp_files.append(temp_name)

    # STEP 2: Rename from temp names to the final clean names
    count = 0
    for i, temp_path in enumerate(temp_files):
        final_name = f"episode_{str(i).zfill(2)}.hdf5"
        final_path = os.path.join(DATA_DIR, final_name)
        
        os.rename(temp_path, final_path)
        print(f"✅ Assigned: {final_name}")
        count += 1

    print(f"\n🎉 Successfully forced {count} files into perfect sequence!")

# --- EXECUTE ---
if __name__ == "__main__":
    force_sequential_rename()
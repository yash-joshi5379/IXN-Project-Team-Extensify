# IXN-Project-Team-Extensify

Repository to store all simulation and hardware-based code.

Goal: create and train a SmolVLA model to automate a robot arm (XArm 7) with a 2-finger gripper to grasp a cylinder and place it into a slot.

## Installation and Setup
1. Clone this repository and go into the project directory
```
git clone https://github.com/yash-joshi5379/IXN-Project-Team-Extensify.git
cd IXN-Project-Team-Extensify/
```

2. Create a virtual environment, activate it, and install all requirements
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Check the ```load_data_test.py``` file runs without errors. It should print the following output.
```
python src\load_data_test.py

                                              action  ... task_index
0  [0.35293713, 0.68267363, -0.07424462, 2.556133...  ...          0
1  [0.35295725, 0.6825197, -0.07424697, 2.5558503...  ...          0
2  [0.35297787, 0.6823274, -0.074249804, 2.55549,...  ...          0
3  [0.35299087, 0.6822012, -0.07425172, 2.5552552...  ...          0
4  [0.35283133, 0.6822953, -0.07299664, 2.5529456...  ...          0

[5 rows x 10 columns]
['action', 'observation.effort', 'observation.force_torque', 'observation.state', 'observation.qvel', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
action                       object
observation.effort           object
observation.force_torque     object
observation.state            object
observation.qvel             object
timestamp                   float32
frame_index                   int64
episode_index                 int64
index                         int64
task_index                    int64
dtype: object
(600, 10)
                                              action  ... task_index
0  [0.35293713, 0.68267363, -0.07424462, 2.556133...  ...          0
1  [0.35295725, 0.6825197, -0.07424697, 2.5558503...  ...          0

[2 rows x 10 columns]
action: shape=(8,), dtype=float32
observation.state: shape=(8,), dtype=float32
observation.effort: shape=(8,), dtype=float32
observation.force_torque: shape=(6,), dtype=float32
observation.qvel: shape=(8,), dtype=float32
```

4. While the venv is active, clone the LeRobot repository
```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
cd ..
```
Note: The -e means "editable," so if you change anything in the LeRobot folder, it updates automatically in your project.

Note: The ```pip install -e .``` command took around 10 mins on my laptop.

5. Uninstall the "headless" version of OpenCV installed from when we installed lerobot, and install the GUI-enabled version:
```
(.venv) ...\IXN-Project-Team-Extensify>pip uninstall opencv-python opencv-python-headless -y
(.venv) ...\IXN-Project-Team-Extensify>pip install opencv-python
```

6. From the main project directory, run the ```check_data.py``` script  with ```EPISODE = episode_000000``` to check the installation works.
```
(.venv) ...\IXN-Project-Team-Extensify>python src\check_data.py
```

It should print the following:
```
Episode 0 has 494 rows of data
Video frame dimensions (height, wdith, channels): (256, 256, 3)
```

7. Install FFmpeg through Command Prompt/PowerShell (used for cropping videos later)
```
(.venv) ...\IXN-Project-Team-Extensify> winget install ffmpeg
```
Note: This is for Windows, on Linux type ```sudo apt install -y ffmpeg``` and on Mac try ```brew install ffmpeg```

8. After installing FFmpeg, close the terminal completely and open a new one, which allows the system to refresh and recognise the new software. In this new terminal, type ```ffmpeg -version```. You should see:
```
ffmpeg version 8.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
built with gcc 15.2.0 (Rev11, Built by MSYS2 project)
...
```



## Preprocessing Data (MUST BE DONE FOR EACH EPISODE)
Before training a model, all videos and datasets must be trimmed upto when the task actually ends (when the cylinder falls into the slot), since any data recorded afterwards is not useful to us.

1. Choose an episode number and camera angle, and adapt the ```VIDEO_PATH``` in ```find_end_frame.py```
```
VIDEO_PATH = "dataset/raw/videos/static_camera/episode_000000.mp4"      # I did episode 0 and static cam
```


2. Run ```find_end_frame.py```. Press/hold the 'd' key to move the video forward, press 'a' to move back a frame, and enter 'q' when you find the perfect end frame for the chosen episode.
```
(.venv) ...\IXN-Project-Team-Extensify>python src\find_end_frame.py

Controls:
 'd' - Next frame
 'a' - Previous frame
 'q' - Quit and print final frame number

---> The task ends at FRAME: 494 <---   # I ran episode 0 and I got frame 494 as the end
```

3. Repeat this process but with the other camera angle. Change the ```VIDEO_PATH``` in ```find_end_frame.py```, and verify you get the same end frame index.
```
VIDEO_PATH = "dataset/raw/videos/onboard_camera/episode_000000.mp4"      # I used onboard cam to verify end index, and it is still frame 494.
```

4. With this end frame index and episode number, put them into ```process_episode.py```, and run ```process_episode.py``` to process the data and videos.
```
# In process_episode.py

# --- Configuration (THIS IS ALL YOU NEED TO CHANGE PER EPSISODE) ---
EPISODE = "episode_000000"   # Episode number (ensure the number has 6 characters) 
END_FRAME = 494              # The exact frame the task finishes (obtained from find_end_frame.py)


# Run the script
(.venv) ...\IXN-Project-Team-Extensify>python src\process_episode.py

# OUTPUT SHOULD BE THIS (for your episode number):
[episode_000000] Trimming parquet data...
Parquet file trimmed.
[episode_000000] Running FFmpeg padding on dataset/raw/videos/static_camera/episode_000000.mp4...
  -> Saved perfectly squared video to dataset/processed/videos/static_camera/episode_000000.mp4
[episode_000000] Running FFmpeg padding on dataset/raw/videos/onboard_camera/episode_000000.mp4...
  -> Saved perfectly squared video to dataset/processed/videos/onboard_camera/episode_000000.mp4

Success! episode_000000 is processed.
```

5. Once this is complete, you should see the dataset and both videos for your episode under ```dataset/processed/data``` and ```dataset/processed/videos``` respectively.

6. For a final check, in ```check_data.py```, edit ```EPISODE = episode_xxxxxxx``` to your episode number and run ```check_data.py```
```
# Run this:

(.venv) ...\IXN-Project-Team-Extensify>python src\check_data.py

# What you should see:

episode_xxxxxx has xxx rows of data                                 # Should match your episode number and the end frame index which you identified.
Video frame dimensions (height, wdith, channels): (256, 256, 3)     # Must be (256, 256, 3) 
```

7. Repeat this whole preprocessing sequence for all your episodes.
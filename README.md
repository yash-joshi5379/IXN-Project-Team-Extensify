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

## Using the LeRobot Visualiser
Once all data has been processed, we must verify it is accurate before training a VLA model. To do this, we will use the LeRobot Visualiser to simultaneously view videos and data plots for each episode.

1. Open the project folder, activate your venv and go into the main project directory.
```
...\IXN-Project-Team-Extensify>.venv\Scripts\activate       # activate venv
(.venv) ...\IXN-Project-Team-Extensify>                     # you should see this
```

2. From the project root directory, run this command:
```
(.venv) ...\IXN-Project-Team-Extensify>lerobot-dataset-viz --repo-id local/cylinder-pick-place --root dataset\processed --mode local --episode-index 0
```

A Rerun window should open and after a few seconds, you should see:
- Both camera feeds (static and wrist cameras) playing simultaneously
- Action data plot (7 arm joints & gripper commands)
- State data plot (7 arm joints & gripper joint angles)
- next.done data plot (Boolean flag that spikes at the final frame of the episode)

Note: After running the command above, many things will be printed in the terminal as well as a progress bar, showing how long it will take to render the plots and video footages. Once this progress bar reaches 100%, all data has fully loaded, so then press the play button in the Rerun window to watch the videos and plots move smoothly over time.
```
100%|████████████████████████████████████████████████████████████████████████████████████| 31/31 [01:17<00:00,  2.50s/it]

# Press the play button in the Rerun window once this reaches 100%
```

Note: If you get this error: **`lerobot-dataset-viz` command not found:**, try this command:
```
python -m lerobot.scripts.lerobot_dataset_viz --repo-id local/cylinder-pick-place --root dataset\processed --mode local --episode-index 0
```

Note: If you get this error: **`ModuleNotFoundError: rerun`:**, try installing rerun again:
```
(.venv) ...\IXN-Project-Team-Extensify> pip install rerun-sdk
```

Note: If the Rerun window opens but shows no data, make sure you run the ```lerobot-dataset-viz``` command from the main project directory ```(.venv) ...\IXN-Project-Team-Extensify>```.

3. Once you have the visualiser working, you can change the episode number. To do this, in the command above, change ```--episode-index 0``` to any number between 0 and 68. E.g.
```
lerobot-dataset-viz --repo-id local/cylinder-pick-place --root dataset\processed --mode local --episode-index 47    # To visualise episode 47
```

4. For each episode, verify the following criteria:
- the videos end with the cylinder in the slot
- the gripper channel in the action plot rises from 0 to 1 during pick-up
- the action and state plots move together with no sudden jumps
- the next.done plot shows only a single spike at the very end and nothing else

5. If the episode meets all 4 requirements, write ```Episode_xxxxxx - valid``` in ```notes.txt```. If not, make a note in ```notes.txt``` and describe which requirement is not met.

6. Repeat for all other allocated episodes.

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
(.venv) ...\IXN-Project-Team-Extensify>python src\visualise_episode.py 0
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

Note: If the Rerun window opens but shows no data, make sure you run the ```python src\visualise_episode.py 0``` command from the main project directory ```(.venv) ...\IXN-Project-Team-Extensify>```.

3. Once you have the visualiser working, you can change the episode number. To do this, in the command above, change ```python src\visualise_episode.py 0``` to any number between 0 and 68. E.g.
```
python src\visualise_episode.py 47    # To visualise episode 47
```

4. For each episode, verify the following criteria:
- the videos end with the cylinder in the slot
- the gripper channel in the action plot rises from 0 to 1 during pick-up
- the action and state plots move together with no sudden jumps
- the next.done plot shows only a single spike at the very end and nothing else

5. If the episode meets all 4 requirements, write ```Episode_xxxxxx - valid``` in ```notes.txt```. If not, make a note in ```notes.txt``` and describe which requirement is not met.

6. Repeat for all other allocated episodes.

## Cleaning Dataset
Using the LeRobot visualiser, we can see if any videos or data plots contain disturbances or sharp jumps. If they do, we do not want to contain this episode in our final dataset for model training, as this erroneous data could worsen our model.

1. To remove an episode, activate your venv, navigate to the project root directory, and run ```src\remove_episode.py x``` where ```x``` is the episode number to be removed. E.g.
```
(.venv) ...\IXN-Project-Team-Extensify>python src\remove_episode.py 0       # to remove episode 0
```

Note: After removing an episode, all subsequent episodes are renumbered, so if you want to remove multiple episodes, remove the **highest numbered episode first** and **lowest numbered episode last**. E.g.
```
python src\remove_episode.py 66   # remove highest numbered episode first
python src\remove_episode.py 17   # then lower ones
python src\remove_episode.py 0    # and lowest numbered episode last
```

## Setup for Accessing UCL Remote GPU Workstations
In order to train our VLA models, we need GPUs for parallel processing and complex computations. For this, we can access remote workstations which have RTX 4070 Ti Super and RTX 4090 GPUs. Here is how to access these workstations:

1. First we need to setup a VPN to access the UCL network. Enter the following URL into a web browser:
```
https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn
```
This webpage contains connection guides and clear instructions for installing the Cisco Anyconnect VPN onto Windows and MacOS devices. Linux installation is possible but not clearly documented on UCL's website, so check ```linux-vpn-install.md``` in this repo for a guide on installing the VPN on Linux devices.

2. Once the VPN is installed, we can access the Remote Workstation Service. Enter the following URL into a web browser:
```
https://tsg.cs.ucl.ac.uk/remote-gpu-workstations/
```

3. Follow the instructions on the webpage to download the UCL CS Root CA certificate and add it to your web browser (Firefox/Chrome).

4. Now activate your Cisco AnyConnect VPN, and then paste this URL: ```https://mydesk.cs.ucl.ac.uk/``` into your web browser. If you are not connected to the VPN at this stage, you will not be able to access this URL.

5. Login to the UCL CS booking system using your UCL Computer Science account username and password **(NOT THE SAME AS YOUR UCL EMAIL AND PASSWORD)**. We got given these CS login details on the first day of our first year. If you cannot remember the details or have lost the details, visit this URL: ```https://tsg.cs.ucl.ac.uk/contact-us/```, and either visit Malet Place in UCL's Bloomsbury Campus or submit the CS Helpdesk Request (much easier).

6. Once you have logged into the UCL CS booking system, you will see a schedule of all GPU workstations and their status (open, reserved, past .etc). You can hover over each workstation name (E.g. bumblebee.cs.ucl.ac.uk) to see which GPU it has.
  
7. To book a session, click on any open (white) cell for your chosen workstation. Give your reservation a title, and adapt the Begin and End times to when you want (maximum reservation time is 72 hours). Click the **Create** button to make the reservation, and you should see your reservation appear on the main schedule.

8. Once your session time has started, you will need to use an SSH tunnel to access your remote GPU workstation.
   
   *Creating an SSH Tunnel on Linux/macOS*

   1. First launch a new terminal on your local laptop/PC, and run the following ssh command, substituting the host name of the machine you booked, and your UCL CS username for $CS_USER. If asked for a password, enter your UCL CS password.
   ```
   ssh -L 8081:<host>.cs.ucl.ac.uk:8443 $CS_USER@knuckles.cs.ucl.ac.uk
   ```
  
   *Creating an SSH Tunnel on Windows*

   1. Launch **WSL** in a new terminal/PowerShell window by running ```wsl``` and then ```cd```. If you do not have WSL installed, simply install it by running ```wsl --install``` in a PowerShell window. Restart your machine after installing WSL to ensure all future terminals have WSL capabilities.
  
   2. Now in your **WSL** terminal, run the following ssh command, substituting the host name of the machine you booked, and your UCL CS username for $CS_USER
   ```
   ssh -L 8081:<host>.cs.ucl.ac.uk:8443 $CS_USER@knuckles.cs.ucl.ac.uk
   ```

      Note: If this doesn't work, open a new PowerShell window and run the same command, without using WSL. If ever asked for a password, enter your UCL CS password.

**If your SSH Tunnel connection is successful, you should see comething like this:**
```
Last login: Mon Apr 13 23:48:58 2026 from 90.254.190.73
>> This machine is running CentOS 7.9
                                                   
>> For all general enquiries, please contact the Helpdesk in 4.07, on
   extn 37280 or e-mail 'request@cs.ucl.ac.uk'

   This machines reboots on the first wednesday of each month
>> Taught students must leave the building before midnight 

** To see this message again type "cat /etc/motd"
...
```

To double check the connection is successful, the terminal should look like ```$CS_USER@knuckles%```, with your CS username instead of $CS_USER. 
To triple check, enter the command ```pwd``` and you should see the following output, with your UCL starting year instead of <year> and your CS username instead of $CS_USER :
```
$CS_USER@knuckles% pwd            # you enter pwd
/cs/student/ug/<year>/$CS_USER    # you should see this with your starting year and CS username instead of <year> and $CS_USER       
```

**IMPORTANT: Keep this terminal window open (the successful SSH Tunnel connection), because this window is the bridge for the SSH connection. If the terminal window, closes, the connection will be lost.**

9. Now that you have remotely connected to the remote GPU workstation via an SSH Tunnel, we can access this connection in VSCode. To do this, open VSCode and install the **Remote - SSH** extension.

10. Open the Command Palette in VSCode by either clicking the Settings icon (bottom right corner of VSCode window) and clicking on the Command Palette option, or by using the keyboard shortcut ```Ctrl+Shift+P```. In the Command Palette, type in and select the option: **Remote-SSH: Open SSH Configuration File**, then select the option which looks like: **.../.ssh/config**. In this config file, enter the following, substituting your CS username instead of $CS_USER and the remote workstation name instead of <host> :
```
Host knuckles
    HostName knuckles.cs.ucl.ac.uk
    User $CS_USER

Host ucl-gpu
    HostName <host>.cs.ucl.ac.uk
    User $CS_USER
    ProxyJump knuckles
```

Then save and close this config file.

11. In the bottom left corner of the VSCode window, you will see the symbol which looks like ```><``` (just under the settings icon). Click this symbol, click the **Connect to Host** option, then click the **ucl-gpu** option. A new VSCode window will appear, where you should enter your CS password in the text prompt area (you may need to enter it twice). After a few seconds, if you see **SSH: ucl-gpu** in the bottom right of the new VSCode window and no errors pop up, the connection is successful.

12. To double check the connection, open a new terminal in the successfully connected VSCode window. You should see ```$CS_USER@<host>%``` in this terminal, with your CS username instead of $CS_USER and the workstation name instead of <host>. Then run the command ```nvidia-smi``` to ensure the GPU is working, and you should see smoething like the following:
```
$CS_USER@<host>% nvidia-smi    # you should see this starting bit in the terminal, and you should run the command 'nvidia smi'
Tue Apr 14 00:36:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 ...    On  |   00000000:01:00.0 Off |                  N/A |
|  0%   33C    P8             10W /  285W |      22MiB /  16376MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           21752      G   /usr/libexec/Xorg                        11MiB |
+-----------------------------------------------------------------------------------------+
```

**This means we have successfully set up a remote connection to a GPU workstation in VSCode!!**

## Setup for Model Training on a UCL Remote GPU Workstation
The next step is to set up the code and environment on the remote workstation, in order to be able to remotely run a training script to run a VLA model.

1. Ensure your remote connection is still intact by running ```pwd``` in a terminal. You should see ```/cs/student/ug/<year>/$CS_USER``` with your UCL starting year instead of ```<year>``` and your CS username instead of ```$CS_USER```.

2. **IMPORTANT - When you reserve time on a remote GPU workstation, you will have scratch space on the machine’s disk in ```/scratch0/$USER/```. Anything in this scratch space will be removed when your sessions ends, so you must ensure that you upload all work on GitHub or download it to your local machine.**

Hence, we will now navigate to the personal scratch space and do everything in this scratch space. To go to your scratch space, paste ```cd /scratch0/$USER``` into your terminal. Then run a ```pwd``` command to ensure you're now in your allocated scratch space.
```
$CS_USER@<host>% pwd                 # check current directory before moving to scratch space
/cs/student/ug/<year>/$CS_USER       # you should see this with your starting year and CS username

$CS_USER@<host>% cd /scratch0/$USER  # use this exact command move to your scratch space
$CS_USER@<host>% pwd                 # now check current directory after moving to scratch space
/scratch0/$CS_USER                   # you should see this with your CS username instead of $CS_USER
```

3. In your scratch space, clone your repository, navigate to the project root directory, and open this repo in a new VSCode window. In this new VSCode window, you should still see the blue **SSH: ucl-gpu** section in the bottom left corner of the window. 
```
$CS_USER@<host>% git clone <repo-url>
$CS_USER@<host>% cd <repo-name>
$CS_USER@<host>% code .
```

4. In this new VSCode window, open a new terminal, and run ```pwd``` to ensure you are in the project root directory: **/scratch0/$CS_USER/repo-name**. You should also see the full project repo structure in the VSCode file explorer on the left side of the VSCode window.

5. Create a venv in the scratch space, by first installing a **standalone, portable Python binary**. We need to do this because the remote workstations only have Python 3.9.25, and we need Python >= 3.12 for this project. Hence, we shall install **Python 3.13.2** :
    1. Navigate back to your personal scratch space by running this in the terminal: ```cd /scratch0/$USER```. Confirm you are in this directory by running ```pwd``` in the terminal.
    
    2. Launch bash by running ```bash``` in the terminal. Your terminal lines should now begin with ```bash-5.1$``` rather than ```$CS_USER@<host>%```.
    
    3. Install **pyenv** by running this command:
    ```
    curl https://pyenv.run | bash

    # You should see this output:

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
    100   270  100   270    0     0   2673      0 --:--:-- --:--:-- --:--:--  2673
    Cloning into '/scratch0/yjoshi/.pyenv'...
    remote: Enumerating objects: 1527, done.
    ...  
    ```

    4. Then run the following 4 commands to add pyenv to the load PATH and to set the cache to scratch permanently.
    ```
    echo 'export PYENV_ROOT="/scratch0/$USER/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'export PIP_CACHE_DIR="/scratch0/$USER/.pip-cache"' >> ~/.bashrc
    ```

    5. Reload your bash by running this command: ```source ~/.bashrc```
    
    6. Intall Python 3.13.2 by running this command: ```pyenv install 3.13.2```
    
    7. Set Python 3.13.2 as the local Python version for this repository with these 2 commands:
    ```
    cd /scratch0/$USER/<repo-name>
    pyenv local 3.13.2
    ```

    8. Verify that these steps worked by checking your Python version, by running ```python --version``` in your bash terminal, and the output should be ```Python 3.13.2```.

6. If not already there, navigate to the project root directory: ```cd /scratch0/$USER/repo-name``` Now create the venv, activate the venv, and install all requirements, all in a bash terminal:
```
bash-5.1$ python -m venv .venv
bash-5.1$ source .venv/bin/activate
(.venv) bash-5.1$ pip install -r requirements.txt 
```

7. Finally follow **instructions 3-6 inclusive** from the first section **(Installation and Setup)**, to run the ```load_data_test.py``` file, clone the Lerobot repo, install its libraries, install the full version of OpenCV, and run the ```check_data.py``` file. Both of these files should run without any errors. You may need to install the Python extension on VSCode for the remote workstation.


## Training a VLA Model on a UCL Remote GPU Workstation
Now that we have set up the venv and are able to run Python scripts on the remote workstation, we move on to training a VLA model. This stage requires some steps on your local machine and some steps on the remote workstation.

### Stage A - On your local machine
1. Create an account at **https://huggingface.co/join**

2. Create an access token at **https://huggingface.co/settings/tokens**, and you MUST give it **Write** permissions. Save the access token value somewhere safe.

3. Ensure you are on your local machine in VSCode (>< symbol under the Settings icon should be grey). If you see the blue **SSH: ucl-gpu** button instead, follow **step 3** in the section **Terminating Remote Workstation Connection** to swtich back to your local machine instead of the remote workstation.

4. Navigate to the project root directory and activate your venv. Then run the command ```pip install --upgrade huggingface_hub``` to ensure you have the latest version of Hugging Face installed in your venv. To check the installation worked, run this command ```python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"``` and the output should look something like this:
```
ModelInfo(id='openai-community/gpt2', author='openai-community', base_models=None, card_data={'base_model': None, 'datasets': None, 'eval_results': None, 'language': 'en', 'library_name': None, 'license': 'mit', 'license_name': None, 'license_link': None, ...
```

5. To login to Hugging Face in VSCode, run the command ```hf auth login```, and paste your access token value when asked for it. Also enter ```y``` when asked to add token as git credential. You should see ```Token is valid (permission: write).```

6. Run the command: ```python src/upload_to_hf.py``` in the terminal to upload the processed dataset to your personal Hugging Face account. To check the upload is successful, there should be no errors in the terminal, and you should see a dataset called ```cylinder-pick-place``` in your profile on the Hugging Face website.

7. Run the command: ```python src/vla-train/download_smolvla_base_weights.py``` to download the SmolVLA base training weights to your local machine. Once downloaded, you should see a **smolvla_base_weights** folder appear in the file explorer, and it should be **grey**. 

### Stage B - On Remote GPU Workstation
1. Connect to your remote workstation host using VSCode. Check the instructions in the section **Setup for Accessing UCL Remote GPU Workstations** if you need help with any steps.

2. In a VSCode terminal, go to your scratch space by running ```cd /scratch0/$USER```, clone the repository, and go into the repo with ```cd repo-name```. Then run ```code .```to see the project repo files on the left hand side of a new VSCode window.

3. Instead of making a venv this time, we will make a **conda env** for training. To do this, we first want to download Miniconda to our scratch space. Run these commands in a new **bash** terminal of the new VSCode window:
```
cd /scratch0/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

4. Run the Miniconda installer with this command: ```bash Miniconda3-latest-Linux-x86_64.sh```. When prompted, press Enter to read the license and type yes to accept the license terms. When asked for the installation location, type ```/scratch0/$USER/miniconda3```. Then when asked to initialise Miniconda3, type yes. Finally to update our bash terminal, run the command ```source ~/.bashrc```. Your bash terminal should now start with ```(base) bash-5.1$ ```.

5. Go into the project root directory with ```cd /scratch0/$USER/repo-name```. Then create the conda env and activate it by using the .yml file:
```
conda env create -f smolvla-gpu-train.yml -y
conda activate smolvla-gpu-train
```

6. Your bash terminal should now begin with ```(smolvla-gpu-train) bash-5.1$ ```. Now run these commands to clone the lerobot libaries required for training SmolVLA:
```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python
cd ..
```

7. Run ```cd ..``` to go back to ```/scratch0/$USER```.  Create a new folder there with this command ```mkdir -p /scratch0/$USER/lerobot_data```. This folder is where the SmolVLA model weights will go.

8. Switch back to your local machine in VSCode using the button under the settings icon. Then run the command below to copy the ```smolvla_base_weights``` which you installed to your local machine, to the remote workstation. In the command, substitute ```$CS_USER``` with your CS username, and substitute ```<host>``` with your remote workstation's name.
On Windows:
```
scp -r -J $CS_USER@knuckles.cs.ucl.ac.uk .\smolvla_base_weights\ $CS_USER@<host>.cs.ucl.ac.uk:/scratch0/$USER/lerobot_data/
```
On Linux/macOS:
```
rsync -avzP -e "ssh -J $CS_USER@knuckles.cs.ucl.ac.uk" ./smolvla_base_weights/ $CS_USER@<host>.cs.ucl.ac.uk:/scratch0/$USER/lerobot_data/smolvla_base_weights
```

9. Switch back to the remote workstation in VSCode, and go to your personal scratch space with ```cd /scratch0/$USER```. Then run this command, substituting your Hugging Face username for ```$HF_USER```:
```
mkdir -p /scratch0/$USER/lerobot_data/lerobot_home/$HF_USER/cylinder-pick-place
```

10. Now run this command, again substituting your Hugging Face username for ```$HF_USER```:
```
cp -a /scratch0/$USER/IXN-Project-Team-Extensify/dataset/processed/. /scratch0/$USER/lerobot_data/lerobot_home/$HF_USER/cylinder-pick-place/
```

11. Go back to the project root directory with ```cd repo-name```. Then run this command, , again substituting your Hugging Face username for ```$HF_USER```:
```
ls -F /scratch0/$USER/lerobot_data/lerobot_home/$HF_USER/cylinder-pick-place/
```
You should see this:
```
data/  meta/  videos/
```



94. Type ```hf auth login``` in the bash terminal with the conda env active, and then hit Enter. Paste in your access token value when prompted to. When asked to add the token as a git credential, type ```N``` as it is not needed now.

95. Type ```wandb login``` in the bash terminal and hit Enter. It will print a URL in the terminal which looks like  **https://wandb.ai/authorize...**. Open this URL in a web browser, sign up for an account (I signed up with a Google account), click on your name in the top right corner of the webpage, go to API keys, and make a new key. Copy your API key, and paste it in the bash terminal. You should see ```Currently logged in as: ... to https://api.wandb.ai.```

96. Create a new **tmux** (terminal multiplexer) session in the bash terminal. This allows you to run the training session on a remote terminal so that the training does not stop if you accidentally close the terminal. To do this, use the command:
```
tmux new -s smolvla_train
```

97. You should see a green bar along the bottom of a new terminal window. Because tmux opens this new terminal window, run ```bash``` to make it a bash terminal, and reactivate your Conda environment with ```conda activate smolvla-gpu-train```. Run ```pip install 'lerobot[dataset]'```.  

98. Before running the training session, do a final check of some important details:
- Check the GPU is active by running ```nvidia-smi```. The **Memory-Usage** should be quite low: mine shows **22MiB /  16376MiB**
- Open config.py to ensure the ```DATASET_REPO_ID``` exactly matches your Hugging Face username and dataset name. Also verify all hyperparameters.
- Verify disk space by running ```df -h .``` inside your ```/scratch/$USER``` directory. You should see a low amount of used storage space (I see 28G) and lots of available storage space (I see 1.5T). You must also see **Mounted on /scratch0** to ensure you are actually in the scratch space and not in your home folder.
- Run ```pwd``` to ensure you are in the project root directory: **/scratch0/$USER/repo-name**. Navigate to this directory if not already there.

99. Once all final checks are done, run the training script with the command below. The ```tee``` command will pipe the output to your screen and save it securely to a file called ```logs.txt```
```
python src/vla-train/train.py 2>&1 | tee logs.txt
``` 

## Terminating Remote Workstation Connection

1. Before your workstation session ends, save the files you want to keep by downloading them to your local machine, or by uploading them to a GitHub branch.

2. Run these commands in VSCode to clear your personal scratch space:
```
cd                                        # go back to your home folder

rm -rf /scratch0/$CS_USER/<repo-name>     # remove the repo in your scratch space

cd /scratch0/$USER                        # after removing, go to your scratch space again
ls -l                                     # check nothing is in your scratch space now
total 0                                   # you should see this if nothing is left in your scratch space
```

2. Close the VSCode terminal, click on the blue **SSH: ucl-gpu** button in the bottom left corner, and choose the **Close Remote Connection** option. Finally, close VSCode, close the terminal window which acted as the SSH bridge between your local machine and the remote workstation, and disconnect from the Cisco VPN. 

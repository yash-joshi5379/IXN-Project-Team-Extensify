import h5py
import cv2
import numpy as np

def visualize_compressed_aloha(file_path):
    print(f"--- Playing: {file_path} ---")
    
    with h5py.File(file_path, 'r') as root:
        actions = root['/action'][()]
        num_frames = actions.shape[0]
        
        cam_names = ['0_femtobolt', '1_gemini330']
        
        print("Controls:")
        print("  [Space] : Play / Pause")
        print("  [b]     : Step Back 1 Frame")
        print("  [f]     : Step Forward 1 Frame")
        print("  [q]     : Quit")

        paused = False
        t = 0
        
        while True:
            # Constrain t so it doesn't crash at the beginning or end
            if t >= num_frames:
                t = num_frames - 1
                paused = True  # Auto-pause at the end of the video
            if t < 0:
                t = 0
            
            # --- 1. Extract and display frame 't' ---
            frames = []
            for cam in cam_names:
                img_bytes = root[f'/observations/images/{cam}'][t]
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Draw the Frame Number
                cv2.putText(img, f"Frame: {t}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw a "PAUSED" indicator so you know when you are in step mode
                if paused:
                    cv2.putText(img, "PAUSED", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                frames.append(img)
            
            if len(frames) == 2:
                combined_frame = np.concatenate(frames, axis=1)
                cv2.imshow("ALOHA Episode Replay", combined_frame)

            # --- 2. Wait for Input ---
            # If paused, wait forever (0). If playing, wait 30ms to create video.
            delay = 0 if paused else 30
            key = cv2.waitKey(delay) & 0xFF
            
            # --- 3. Process Input ---
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                paused = not paused
                # If unpausing at the very end of the video, loop back to the start
                if not paused and t == num_frames - 1:
                    t = 0 
            elif key == ord('b'):  # 'b' for Back
                t -= 1
                paused = True # Force pause so the video doesn't fly past
            elif key == ord('f'):  # 'f' for Forward
                t += 1
                paused = True # Force pause
            else:
                # If the video is playing and no control keys were pressed, advance time
                if not paused:
                    t += 1

    cv2.destroyAllWindows()



# ONLY CHANGE THE FILE PATH, NOTHING ELSE
visualize_compressed_aloha("processed_aloha_dataset/episode_45.hdf5")
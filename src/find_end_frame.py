import cv2

# Change this to whatever episode and camera angle you are checking
# WHEN CHANGING CAMERA ANGLE, JUST CHANGE onboard_camera TO static_camera AND VICE VERSA
VIDEO_PATH = "dataset/raw/videos/onboard_camera/episode_000001.mp4" 

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

print("Controls:")
print(" 'd' - Next frame")
print(" 'a' - Previous frame")
print(" 'q' - Quit and print final frame number")

while True:
    # Jump to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print("End of video reached.")
        break
        
    # Write the frame number directly onto the video in red text
    cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Annotator", frame)
    
    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('d'):   # Press 'd' to go forward
        frame_idx += 1
    elif key == ord('a'): # Press 'a' to go backward
        frame_idx = max(0, frame_idx - 1) # Prevents going into negative numbers
    elif key == ord('q'): # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n---> The task ends at FRAME: {frame_idx} <---")
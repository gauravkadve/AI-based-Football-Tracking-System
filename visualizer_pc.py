# slider

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from annotator import draw_pitch, draw_points_on_pitch, draw_player_ids_on_pitch
from config import SoccerPitchConfiguration

# Initialize the soccer pitch configuration and colors
CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']  # Adjust or extend as needed

# Hardcoded CSV path
CSV_PATH = r"D:\VIT\4th year\8th Sem\Main Project\radar_points_csv_final.csv"  # Replace with your actual CSV file path

# Global variables
max_frame = 0
df = None
window_name = "Soccer Pitch Visualization"

def plot_frame(frame_idx: int) -> np.ndarray:
    global df
    
    # Filter the points corresponding to the given frame index
    df_frame = df[df['frame'] == frame_idx]
    if df_frame.empty:
        print(f"No data found for frame {frame_idx}.")
        # Create an empty pitch if no data is available
        return draw_pitch(config=CONFIG)
    
    # Create a base pitch image using the soccer configuration
    pitch = draw_pitch(config=CONFIG)
    
    # For each team (or class) present in the data, plot the points and player IDs
    for team_id in sorted(df_frame['team_id'].unique()):
        # Filter data for the team
        df_team = df_frame[df_frame['team_id'] == team_id]
        # Get the (x, y) coordinates as a NumPy array
        points = df_team[['x', 'y']].to_numpy()
        # Get the player IDs (as strings)
        player_ids = df_team['player_id'].astype(str).tolist()
        
        # Draw the points onto the pitch using the team color
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=points,
            face_color=sv.Color.from_hex(COLORS[int(team_id) % len(COLORS)]),
            radius=20,
            pitch=pitch
        )
        
        # Draw the player IDs near their respective points
        pitch = draw_player_ids_on_pitch(
            config=CONFIG,
            xy=points,
            player_ids=player_ids,
            text_color=sv.Color.WHITE,
            bg_color=sv.Color.from_hex(COLORS[int(team_id) % len(COLORS)]),
            font_scale=0.8,
            font_thickness=2,
            text_offset=(0, -25),
            pitch=pitch
        )
    
    # Add frame number text to the image
    cv2.putText(
        pitch, 
        f"Frame: {frame_idx}", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2, 
        cv2.LINE_AA
    )
        
    return pitch

def on_trackbar_change(value):
    # Plot the frame corresponding to the slider value
    frame = plot_frame(value)
    # Display the resulting frame
    cv2.imshow(window_name, frame)

def main():
    global df, max_frame
    
    try:
        # Load the CSV data
        df = pd.read_csv(CSV_PATH)
        
        # Get the maximum frame number for the slider
        max_frame = int(df['frame'].max())
        min_frame = int(df['frame'].min())
        
        # Create a window and a trackbar/slider
        cv2.namedWindow(window_name)
        cv2.createTrackbar('Frame', window_name, min_frame, max_frame, on_trackbar_change)
        
        # Initialize with the first frame
        initial_frame = min_frame
        on_trackbar_change(initial_frame)
        
        print("Use the slider to navigate through frames. Press 'q' to quit.")
        
        # Wait for a key press to exit
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            # Add keyboard navigation for frames
            elif key == ord('a') or key == 81:  # 'a' or left arrow
                current_frame = cv2.getTrackbarPos('Frame', window_name)
                if current_frame > min_frame:
                    cv2.setTrackbarPos('Frame', window_name, current_frame - 1)
            
            elif key == ord('d') or key == 83:  # 'd' or right arrow
                current_frame = cv2.getTrackbarPos('Frame', window_name)
                if current_frame < max_frame:
                    cv2.setTrackbarPos('Frame', window_name, current_frame + 1)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

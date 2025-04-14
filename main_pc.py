# Combined Team Classification and Radar with CSV Export

import cv2
import csv
import numpy as np
import supervision as sv
from enum import Enum
from typing import Iterator, List, Dict, Any
from tqdm import tqdm
from ultralytics import YOLO

from annotator import draw_pitch, draw_player_ids_on_pitch, draw_points_on_pitch
from ball import BallTracker, BallAnnotator
from team import TeamClassifier
from view import ViewTransformer
from config import SoccerPitchConfiguration

# Path configurations
PLAYER_DETECTION_MODEL_PATH = r"D:\VIT\4th year\8th Sem\Main Project\player.pt"
PITCH_DETECTION_MODEL_PATH = r"D:\VIT\4th year\8th Sem\Main Project\keypoint_tracker.pt"
BALL_DETECTION_MODEL_PATH = r"D:\VIT\4th year\8th Sem\Main Project\ball.pt"

# Class IDs for classification
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# constants
STRIDE = 60
CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

tracking_data = []

# Get crops from the frame based on detected bounding boxes by the detection models
def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

#  Resolve team IDs for goalkeepers. ID given with respect to the proximity of the team near it
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    if len(goalkeepers) == 0 or len(players) == 0:
        return np.array([])
        
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    team_0_indices = np.where(players_team_id == 0)[0]
    team_1_indices = np.where(players_team_id == 1)[0]
    
    if len(team_0_indices) == 0 or len(team_1_indices) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    
    team_0_centroid = players_xy[team_0_indices].mean(axis=0)
    team_1_centroid = players_xy[team_1_indices].mean(axis=0)
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(goalkeepers_team_id)

# RADAR function
def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    tracker_ids: List[int],
    frame_index: int
) -> np.ndarray:
    if len(keypoints.xy) == 0 or len(detections) == 0:          # base condition incase no bounding boxes are detected
        return draw_pitch(config=CONFIG)
    
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if not np.any(mask) or np.sum(mask) < 4:            # Need at least 4 points for transformation
        return draw_pitch(config=CONFIG)
    
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)
    
    radar = draw_pitch(config=CONFIG)
    
    # Process each team
    for team_id in range(4):                            # 0, 1 for teams, 2 for referee, 3 for others
        team_indices = np.where(color_lookup == team_id)[0]
        if len(team_indices) > 0:
            team_points = transformed_xy[team_indices]
            team_tracker_ids = [tracker_ids[i] for i in team_indices]
            
            # Draw points on radar
            radar = draw_points_on_pitch(
                config=CONFIG,
                xy=team_points,
                face_color=sv.Color.from_hex(COLORS[team_id]),
                radius=20,
                pitch=radar
            )
            
            # Log tracking data to export it to CSV 
            for point, player_id in zip(team_points, team_tracker_ids):
                if player_id is not None:
                    tracking_data.append({
                        "frame": frame_index,
                        "team_id": int(team_id),
                        "player_id": int(player_id),
                        "x": float(point[0]),
                        "y": float(point[1])
                    })
            
            # Draw player IDs on radar
            valid_ids = []
            valid_points = []
            for point, player_id in zip(team_points, team_tracker_ids):
                if player_id is not None:
                    valid_ids.append(str(player_id))
                    valid_points.append(point)
            
            if valid_points:
                radar = draw_player_ids_on_pitch(
                    config=CONFIG,
                    xy=np.array(valid_points),
                    player_ids=valid_ids,
                    text_color=sv.Color.WHITE,
                    bg_color=sv.Color.from_hex(COLORS[team_id]),
                    font_scale=0.8,
                    font_thickness=2,
                    text_offset=(0, -25),
                    pitch=radar
                )
    
    return radar

# Run combined team classification and radar mode
def run_combined_mode(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    global tracking_data
    
    # models initialization
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    
    # crops for team classifier
    print("Collecting player crops for team classification...")
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    
    # Train team classifier
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    # Initialize trackers                               ByteTrack Tracking algorithm
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # Ball detection callback
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    ball_slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640))
    
    # Process video frames
    print("Processing video frames...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    frame_index = 0
    
    for frame in frame_generator:
        # Using Pitch detection model for radar view
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        
        # using Player detection model and tracking
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)
        
        # Classifing teams
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        
        if len(players) > 0:
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)
        else:
            players_team_id = np.array([])
        
        # goalkeepers detection and resolving
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        if len(goalkeepers) > 0 and len(players) > 0 and len(players_team_id) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([])
        
        # similarly for referees
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        
        # Merge everything together                         (detections)
        team_detections = sv.Detections.merge([players, goalkeepers, referees])
        
        # color lookup array                                    (integer types)
        color_lookup = np.concatenate([
            players_team_id.astype(int),
            goalkeepers_team_id.astype(int),
            np.full(len(referees), REFEREE_CLASS_ID, dtype=int)
        ]) if len(team_detections) > 0 else np.array([], dtype=int)
        
        # Detecting the football
        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)
        
        # annotate the frame with team classification
        annotated_frame = frame.copy()
        
        if len(team_detections) > 0:
            labels = [str(tracker_id) for tracker_id in team_detections.tracker_id]
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, team_detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, team_detections, labels, custom_color_lookup=color_lookup)
        
        # Add ball detections to the annotated frames
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        
        # Generate radar view
        if len(team_detections) > 0:
            radar = render_radar(
                team_detections, 
                keypoints, 
                color_lookup, 
                team_detections.tracker_id, 
                frame_index
            )
            
            # Display 
            cv2.imshow("Radar View", radar)
        
        frame_index += 1
        yield annotated_frame

# saving the data to csv file
def save_tracking_data_csv(output_path: str) -> None:
    if not tracking_data:
        print("No tracking data to save.")
        return
    
    keys = tracking_data[0].keys()
    with open(output_path, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(tracking_data)
    
    print(f"Tracking data saved to {output_path}")

# main funtion to run the program
def main(source_video_path: str, target_video_path: str, csv_output_path: str, device: str) -> None:
    global tracking_data
    tracking_data = []  # Clear previous data
    
    # Get video info
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    width = int(video_info.width)
    height = int(video_info.height)
    fps = video_info.fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_video_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_generator = run_combined_mode(source_video_path=source_video_path, device=device)
    
    for frame in frame_generator:
        out.write(frame)
        
        # Display frame (optional)
        cv2.imshow("Combined Team Classification and Radar", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Release resources
    out.release()
    cv2.destroyAllWindows()
    
    # Save tracking data to CSV
    save_tracking_data_csv(csv_output_path)
    
    print(f"Video processing complete. Output saved to {target_video_path}")

if __name__ == '__main__':
    source_video_path = r"D:\VIT\4th year\8th Sem\Main Project\08fd33_4.mp4"
    target_video_path = rf"D:\VIT\4th year\8th Sem\Main Project\result.mp4"
    csv_output_path = r"D:\VIT\4th year\8th Sem\Main Project\result.csv"
    device = 'cuda'  # Use 'cpu' if CUDA is not available
    
    # Run the combined mode
    main(
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        csv_output_path=csv_output_path,
        device=device
    )

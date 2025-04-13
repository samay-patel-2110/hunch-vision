"""
    Hunch Detection Module ( YOLO BASED )

    This module provides functions to detect hunch posture using YOLO model.
    It can be imported into other projects or run as a standalone script with webcam input.

    The module is structured with these main classes:
    - PostureDetectorYOLO: Handles hunch detection using YOLO model
"""

import cv2
import numpy as np
import math
import time
from collections import deque
from ultralytics import YOLO
import pygame
import threading

class PostureDetectorYOLO:
    def __init__(self, model_path="./yolov11n-pose.pt"):
        print("Initializing PostureDetectorYOLO...")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize audio for cue
        pygame.mixer.init()
        try:
            self.cue_sound = pygame.mixer.Sound("cue.mp3")
            print("Successfully loaded audio cue")
        except Exception as e:
            print(f"Error loading audio cue: {e}")
            self.cue_sound = None
            
        # Audio control variables
        self.last_audio_time = 0
        self.audio_cooldown = 5  # seconds between audio cues
        self.was_hunched = False  # Track previous state

        # Joint dictionary for YOLO keypoints
        self.joints_dict = {
            0: "Nose",
            1: "Right Eye",
            2: "Left Eye",
            3: "Right Ear",
            4: "Left Ear",
            5: "Right Shoulder",
            6: "Left Shoulder",
            7: "Right Elbow",
            8: "Left Elbow",
            9: "Right Wrist",
            10: "Left Wrist",
            11: "Right Hip",
            12: "Left Hip",
            13: "Right Knee",
            14: "Left Knee",
            15: "Right Ankle",
            16: "Left Ankle"
        }
        
        # Upper body keypoints we're interested in
        self.upper_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        # Angle thresholds for posture classification
        self.hunched_threshold = 1.08  # Shoulder drop ratio threshold
        
        # Status variables
        self.posture_history = deque(maxlen=7)  # Increased history size for smoother output
        
        # Calibration variables
        self.calibrated = False
        self.calibration_in_progress = False
        self.calibration_start_time = 0
        self.calibration_duration = 5  # seconds
        self.calibration_samples = []
        self.baseline_neck_angle = 0
        self.baseline_shoulder_position = None
        self.baseline_shoulder_distance = None  # Added for scale normalization
        
        # Shoulder drop detection variables
        self.baseline_shoulder_drop_distance = None
        self.shoulder_track_point = None
        self.shoulder_drop_threshold = 1.08  # Threshold for detecting shoulder drop (ratio)
        self.shoulder_drop_history = deque(maxlen=10)
        self.shoulder_drop_detected = False
        
        # Colors for visualization
        self.colors = {
            'upright': (0, 255, 0),  # Green
            'hunched': (0, 0, 255),  # Red
            'midpoint': (255, 255, 0),  # Yellow
            'angle': (255, 0, 255),  # Magenta
            'reference': (0, 255, 255),  # Cyan
            'joint': (255, 165, 0),    # Orange
            'bbox': (255, 255, 255),    # White
            'label': (50, 205, 50),      # Lime green for labels
            'track_point': (0, 140, 255)  # Orange for tracking point
        }
        
        # Confidence threshold for keypoints
        self.confidence_threshold = 0.45  # Slightly reduced to capture more keypoints
        
        # Initialize FPS calculation
        self.prev_frame_time = 0
        self.fps = 0
        
        # Initialize analytics
        self.hunch_duration = 0
        self.start_hunch_time = None
        self.posture_stats = {
            "upright_time": 0,
            "hunched_time": 0,
            "total_time": 0
        }
        self.last_update_time = time.time()

        # Shoulder drop tracking
        self.prev_frame_gray = None
        self.track_point = None
        self.track_point_initial = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        

    def calculate_shoulder_track_point(self, keypoints):
        """Calculate a point below mid shoulder at 1/4 the distance of shoulder width"""
        if 5 not in keypoints or 6 not in keypoints:
            return None
            
        # Get shoulder points
        right_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
        left_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
        
        # Calculate mid shoulder
        mid_shoulder = (right_shoulder + left_shoulder) / 2
        
        # Calculate shoulder width
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Calculate point below mid shoulder at 1/4 shoulder width distance
        track_point = (int(mid_shoulder[0]), int(mid_shoulder[1] + shoulder_width * 0.25))
        
        return track_point, mid_shoulder
    
    def start_calibration(self):
        """Start calibration process to establish baseline posture"""
        print("Starting calibration... Please sit upright for 5 seconds.")
        self.calibration_in_progress = True
        self.calibration_samples = []
        self.calibration_start_time = time.time()
        self.calibrated = False
        
        # Reset analytics when recalibrating
        self.posture_stats = {
            "upright_time": 0,
            "hunched_time": 0,
            "total_time": 0
        }
        self.hunch_duration = 0
        self.start_hunch_time = None

    def update_calibration(self, keypoints):
        """Update calibration with new measurements"""
        if not self.calibration_in_progress:
            return False
            
        # Get neck angle
        neck_angle = self.calculate_neck_angle(keypoints)
        
        # Get shoulder midpoint and distance
        shoulder_mid = self.calculate_shoulder_midpoint(keypoints)
        shoulder_distance = self.calculate_shoulder_distance(keypoints)
        
        # ADD: Get shoulder track point
        shoulder_data = self.calculate_shoulder_track_point(keypoints)
        
        if neck_angle is not None and shoulder_mid is not None and shoulder_distance is not None and shoulder_data is not None:
            track_point, mid_shoulder = shoulder_data
            
            # Calculate vertical distance between mid shoulder and track point
            vertical_distance = abs(track_point[1] - mid_shoulder[1])
            
            self.calibration_samples.append((
                neck_angle, 
                shoulder_mid, 
                shoulder_distance, 
                vertical_distance,  # ADD: Store vertical distance
                track_point  # ADD: Store track point
            ))
        
        # Check if calibration duration has elapsed
        elapsed_time = time.time() - self.calibration_start_time
        if elapsed_time >= self.calibration_duration:
            if len(self.calibration_samples) > 5:  # Ensure we have at least 5 samples
                # Calculate average neck angle
                angles = [sample[0] for sample in self.calibration_samples]
                self.baseline_neck_angle = sum(angles) / len(angles)
                
                # Calculate average shoulder position
                shoulder_positions = [sample[1] for sample in self.calibration_samples]
                avg_shoulder_x = sum(pos[0] for pos in shoulder_positions) / len(shoulder_positions)
                avg_shoulder_y = sum(pos[1] for pos in shoulder_positions) / len(shoulder_positions)
                self.baseline_shoulder_position = (avg_shoulder_x, avg_shoulder_y)
                
                # Calculate average shoulder distance
                distances = [sample[2] for sample in self.calibration_samples]
                self.baseline_shoulder_distance = sum(distances) / len(distances)
                
                # ADD: Calculate baseline for shoulder drop distance
                drop_distances = [sample[3] for sample in self.calibration_samples]
                self.baseline_shoulder_drop_distance = sum(drop_distances) / len(drop_distances)
                
                # ADD: Set initial track point (average of calibration points)
                track_points = [sample[4] for sample in self.calibration_samples]
                avg_x = sum(pt[0] for pt in track_points) / len(track_points)
                avg_y = sum(pt[1] for pt in track_points) / len(track_points)
                self.shoulder_track_point = (int(avg_x), int(avg_y))
                
                self.calibrated = True
                self.calibration_in_progress = False
                print(f"Calibration complete. Baseline neck angle: {self.baseline_neck_angle:.2f}°")
                print(f"Baseline shoulder position: {self.baseline_shoulder_position}")
                print(f"Baseline shoulder distance: {self.baseline_shoulder_distance:.2f}")
                print(f"Baseline shoulder drop distance: {self.baseline_shoulder_drop_distance:.2f}")
                
                # Reset posture stats at calibration
                self.posture_stats = {
                    "upright_time": 0,
                    "hunched_time": 0,
                    "total_time": 0
                }
                self.last_update_time = time.time()
                return True
            else:
                print("Calibration failed: Not enough samples collected. Please try again.")
                self.calibration_in_progress = False
                return False
        
        return False
    
    def get_keypoints(self, keypoints_data):
        """Extract keypoints from YOLO detection"""
        keypoints = {}
        
        # Convert to dictionary for easier access
        for i in range(len(keypoints_data[0])):
            x, y, conf = keypoints_data[0][i]
            if conf > self.confidence_threshold and i in self.upper_body_keypoints:
                keypoints[i] = (float(x), float(y), float(conf))
        
        return keypoints
    
    def calculate_neck_angle(self, keypoints):
        """Calculate the forward lean angle based on shoulders and nose"""
        if 5 not in keypoints or 6 not in keypoints or 0 not in keypoints:
            return None
            
        # Get relevant keypoints
        right_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
        left_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
        nose = np.array([keypoints[0][0], keypoints[0][1]])
        
        # Calculate shoulder midpoint
        shoulder_mid = (right_shoulder + left_shoulder) / 2
        
        # Calculate angle between vertical line from shoulders and line to nose
        dx = nose[0] - shoulder_mid[0]
        dy = nose[1] - shoulder_mid[1]
        
        # Calculate angle (negative to match convention where forward lean is positive)
        angle = -math.degrees(math.atan2(dx, -dy))
        
        return angle
    
    def calculate_shoulder_midpoint(self, keypoints):
        """Calculate midpoint between shoulders"""
        if 5 not in keypoints or 6 not in keypoints:
            return None
            
        right_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
        left_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
        
        return ((right_shoulder + left_shoulder) / 2).tolist()
    
    def calculate_shoulder_distance(self, keypoints):
        """Calculate distance between shoulders for scale normalization"""
        if 5 not in keypoints or 6 not in keypoints:
            return None
            
        right_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
        left_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
        
        return np.linalg.norm(right_shoulder - left_shoulder)
    
    def track_shoulder_point(self, frame_gray):
        """Track the point below mid shoulder using optical flow"""
        if self.prev_frame_gray is None or self.track_point is None:
            return False
        
        # Convert tracking point to proper format for optical flow
        p0 = np.array([[self.track_point[0], self.track_point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray,
            frame_gray,
            p0,
            None,
            **self.lk_params
        )
        
        # Update if tracking was successful
        if st[0][0] == 1:
            self.track_point = (int(p1[0][0][0]), int(p1[0][0][1]))
            self.prev_frame_gray = frame_gray.copy()
            return True
        
        return False
    
    def detect_shoulder_drop(self, keypoints):
        """Detect if shoulders are dropping based on tracked point"""
        if not self.calibrated or self.baseline_shoulder_drop_distance is None:
            return False, 0
            
        # Calculate current shoulder track point
        shoulder_data = self.calculate_shoulder_track_point(keypoints)
        if shoulder_data is None:
            return False, 0
        
        # Get current track point and mid shoulder
        current_track_point, current_mid_shoulder = shoulder_data
        
        # Calculate current vertical distance
        current_distance = abs(current_track_point[1] - current_mid_shoulder[1])
        
        # Calculate ratio to baseline
        drop_ratio = current_distance / self.baseline_shoulder_drop_distance
        
        # Add to history for smoothing
        self.shoulder_drop_history.append(drop_ratio)
        
        # Get average ratio from history
        avg_ratio = sum(self.shoulder_drop_history) / len(self.shoulder_drop_history)
        
        # Update shoulder track point for visualization
        self.shoulder_track_point = current_track_point
        
        # Detect drop if ratio exceeds threshold
        self.shoulder_drop_detected = avg_ratio > self.shoulder_drop_threshold
        
        return self.shoulder_drop_detected, avg_ratio
    
    def create_upper_body_bbox(self, keypoints, frame_height, frame_width):
        """Create a bounding box for the upper body (neck and below)"""
        if not (5 in keypoints and 6 in keypoints):
            return None
        
        # Get shoulder points
        rs = np.array([keypoints[5][0], keypoints[5][1]])
        ls = np.array([keypoints[6][0], keypoints[6][1]])
        
        # Get hip points if available, otherwise estimate
        if 11 in keypoints and 12 in keypoints:
            rh = np.array([keypoints[11][0], keypoints[11][1]])
            lh = np.array([keypoints[12][0], keypoints[12][1]])
        else:
            # Estimate hip position based on shoulders
            shoulder_distance = np.linalg.norm(rs - ls)
            rh = rs + np.array([0, shoulder_distance * 1.5])
            lh = ls + np.array([0, shoulder_distance * 1.5])
        
        # Calculate bbox with padding
        shoulder_distance = np.linalg.norm(rs - ls)
        padding_x = shoulder_distance * 0.5
        padding_y_top = shoulder_distance * 0.2
        padding_y_bottom = shoulder_distance * 0.3
        
        min_x = max(0, int(min(rs[0], ls[0]) - padding_x))
        max_x = min(frame_width, int(max(rs[0], ls[0]) + padding_x))
        
        # Start from neck/shoulders with padding
        min_y = max(0, int(min(rs[1], ls[1]) - padding_y_top))
        max_y = min(frame_height, int(max(rh[1], lh[1]) + padding_y_bottom))
        
        return (min_x, min_y, max_x, max_y)
    
    def classify_posture(self, shoulder_drop):
        """Classify posture based solely on shoulder drop detection"""
        if not self.calibrated:
            return "Unknown"
            
        # Update stats tracking
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.posture_stats["total_time"] += dt
        
        if shoulder_drop:
            # Track continuous hunching
            if self.start_hunch_time is None:
                self.start_hunch_time = current_time
            
            # Update hunched time statistics
            self.posture_stats["hunched_time"] += dt
            return "Hunched"
        else:
            # Reset hunch time if posture corrected
            if self.start_hunch_time is not None:
                self.hunch_duration = 0
                self.start_hunch_time = None
            
            # Update upright time statistics
            self.posture_stats["upright_time"] += dt
            return "Upright"
    
    def draw_joint_connections(self, frame, keypoints, w, h):
        """Draw connections between joints"""
        connections = [
            (0, 1), (0, 2),  # Nose to eyes
            (1, 3), (2, 4),  # Eyes to ears
            (5, 6),  # Shoulders
            (5, 7), (7, 9),  # Right arm
            (6, 8), (8, 10),  # Left arm
            (5, 11), (6, 12),  # Shoulders to hips
            (11, 12)  # Hips
        ]
        
        for conn in connections:
            if conn[0] in keypoints and conn[1] in keypoints:
                pt1 = (int(keypoints[conn[0]][0]), int(keypoints[conn[0]][1]))
                pt2 = (int(keypoints[conn[1]][0]), int(keypoints[conn[1]][1]))
                cv2.line(frame, pt1, pt2, self.colors['joint'], 2)
    
    def draw_keypoints_info(self, frame, keypoints, w, h):
        """Draw keypoints and their information"""
        # Draw connections first (so they're behind the points)
        self.draw_joint_connections(frame, keypoints, w, h)
        
        # Draw joints with labels
        for kp_id, (x, y, conf) in keypoints.items():
            x_px, y_px = int(x), int(y)
            
            # Only draw keypoints above confidence threshold
            if conf > self.confidence_threshold:
                # Create a filled circle with border for better visibility
                cv2.circle(frame, (x_px, y_px), 6, (0, 0, 0), -1)  # Black border
                cv2.circle(frame, (x_px, y_px), 4, self.colors['joint'], -1)  # Colored center
                
                # Draw joint name with better visibility
                joint_name = self.joints_dict.get(kp_id, f"Joint {kp_id}")
                
                # Create background for text for better readability
                text_size = cv2.getTextSize(joint_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    frame, 
                    (x_px + 5, y_px - 15), 
                    (x_px + text_size[0] + 10, y_px + 5),
                    (0, 0, 0), 
                    -1
                )
                
                cv2.putText(
                    frame, 
                    joint_name, 
                    (x_px + 7, y_px), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    self.colors['label'], 
                    1
                )
        
        return frame
    
    def update_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time
        
        # Smooth FPS
        self.fps = 0.9 * self.fps + 0.1 * fps if self.fps > 0 else fps
        return self.fps
    
    def draw_analytics(self, frame, posture):
        """Draw posture analytics on frame"""
        h, w, _ = frame.shape
        
        # Calculate percentages
        total = max(1, self.posture_stats["total_time"])
        upright_pct = (self.posture_stats["upright_time"] / total) * 100
        hunched_pct = (self.posture_stats["hunched_time"] / total) * 100
        
        # Create analytics panel
        panel_height = 120
        panel_y = h - panel_height
        
        # Draw semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw title
        cv2.putText(frame, "POSTURE ANALYTICS", (20, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw time bars
        bar_width = w - 40
        bar_height = 20
        bar_x = 20
        bar_y = panel_y + 40
        
        # Total bar (background)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Upright time (green)
        upright_width = int((upright_pct / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + upright_width, bar_y + bar_height), 
                     self.colors['upright'], -1)
        
        # Add percentage text
        cv2.putText(frame, f"Upright: {upright_pct:.1f}%", (bar_x + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Hunched: {hunched_pct:.1f}%", (bar_x + bar_width - 150, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add current streak info
        if posture == "Hunched" and self.start_hunch_time is not None:
            hunch_duration = time.time() - self.start_hunch_time
            cv2.putText(frame, f"Current Hunched Duration: {hunch_duration:.1f}s", 
                       (bar_x, bar_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['hunched'], 2)
            
            if hunch_duration > 30:  # Warning after 30 seconds of hunching
                cv2.putText(frame, "WARNING: Consider adjusting your posture!", 
                           (bar_x, bar_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 0, 255), 2)
        
        return frame
    
    def play_audio_cue(self):
        """Play audio cue in a separate thread to avoid blocking the main processing"""
        if self.cue_sound is None:
            return
            
        # Play in a separate thread to avoid blocking
        def play_sound():
            self.cue_sound.play()
            
        threading.Thread(target=play_sound).start()
    
    def process_frame(self, frame):
        """Process a single frame to detect posture"""
        h, w, _ = frame.shape
        
        # Convert to grayscale for optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update FPS
        fps = self.update_fps()
        
        # Run YOLO model on frame
        results = self.model(frame, verbose=False)
        
        # Default values
        posture = "Unknown"
        neck_angle = None
        shoulder_mid = None
        upper_body_bbox = None
        shoulder_drop = False
        drop_ratio = 0
        
        # Process results
        if results and len(results) > 0 and results[0].keypoints is not None:
            # Extract keypoints
            keypoints_data = results[0].keypoints.data
            
            if len(keypoints_data) > 0:
                # Get keypoints in dictionary format
                keypoints = self.get_keypoints(keypoints_data)
                
                # Update calibration if in progress
                if self.calibration_in_progress:
                    self.update_calibration(keypoints)
                
                # Calculate neck angle
                neck_angle = self.calculate_neck_angle(keypoints)
                
                # Calculate shoulder midpoint
                shoulder_mid = self.calculate_shoulder_midpoint(keypoints)
                
                # Detect shoulder drop if calibrated
                if self.calibrated:
                    shoulder_drop, drop_ratio = self.detect_shoulder_drop(keypoints)
                
                # Create upper body bounding box
                upper_body_bbox = self.create_upper_body_bbox(keypoints, h, w)
                
                # MODIFY: Classify posture based only on shoulder drop
                if self.calibrated:
                    posture = self.classify_posture(shoulder_drop)
                    
                    # Add to history for smoothing
                    self.posture_history.append(posture)
                    
                    # Get majority vote from history
                    if len(self.posture_history) >= 3:
                        upright_count = self.posture_history.count("Upright")
                        hunched_count = self.posture_history.count("Hunched")
                        posture = "Upright" if upright_count >= hunched_count else "Hunched"
                    
                    # Play audio cue when transitioning from upright to hunched
                    current_time = time.time()
                    if posture == "Hunched" and not self.was_hunched:
                        if current_time - self.last_audio_time > self.audio_cooldown:
                            self.play_audio_cue()
                            self.last_audio_time = current_time
                    
                    # Update previous state
                    self.was_hunched = (posture == "Hunched")
                
                # Draw upper body bounding box if available
                if upper_body_bbox:
                    cv2.rectangle(frame, 
                                 (upper_body_bbox[0], upper_body_bbox[1]), 
                                 (upper_body_bbox[2], upper_body_bbox[3]), 
                                 self.colors['bbox'], 2)
                    
                    # Add semi-transparent background for label
                    label_bg = frame.copy()
                    cv2.rectangle(label_bg, 
                                 (upper_body_bbox[0], upper_body_bbox[1] - 30), 
                                 (upper_body_bbox[0] + 160, upper_body_bbox[1]), 
                                 (0, 0, 0), -1)
                    cv2.addWeighted(label_bg, 0.6, frame, 0.4, 0, frame)
                    
                    cv2.putText(frame, "UPPER BODY REGION", 
                               (upper_body_bbox[0] + 10, upper_body_bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['bbox'], 2)
                
                # ADD: Visualize shoulder drop tracking
                if self.calibrated and shoulder_mid is not None and self.shoulder_track_point is not None:
                    # Get shoulder data
                    shoulder_data = self.calculate_shoulder_track_point(keypoints)
                    if shoulder_data is not None:
                        current_track_point, current_mid_shoulder = shoulder_data
                        
                        # Draw mid shoulder point
                        mid_x, mid_y = int(current_mid_shoulder[0]), int(current_mid_shoulder[1])
                        cv2.circle(frame, (mid_x, mid_y), 7, (0, 0, 0), -1)  # Black outline
                        cv2.circle(frame, (mid_x, mid_y), 5, self.colors['midpoint'], -1)  # Yellow fill
                        cv2.putText(frame, "Mid Shoulder", (mid_x + 10, mid_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['midpoint'], 2)
                        
                        # Draw track point
                        track_x, track_y = current_track_point
                        cv2.circle(frame, (track_x, track_y), 7, (0, 0, 0), -1)  # Black outline
                        cv2.circle(frame, (track_x, track_y), 5, self.colors['track_point'], -1)  # Orange fill
                        cv2.putText(frame, "Track Point", (track_x + 10, track_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['track_point'], 2)
                        
                        # Draw line between mid shoulder and track point
                        cv2.line(frame, (mid_x, mid_y), (track_x, track_y), 
                                self.colors['track_point'], 2)
                        
                        # Show vertical distance and ratio
                        if self.baseline_shoulder_drop_distance is not None:
                            current_distance = abs(track_y - mid_y)
                            ratio_text = f"Drop Ratio: {drop_ratio:.2f}x"
                            dist_text = f"Distance: {current_distance:.1f} px"
                            
                            # Add background for text
                            text_pos = (mid_x - 100, mid_y + 60)
                            text_size = cv2.getTextSize(ratio_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            
                            cv2.rectangle(frame, 
                                        (text_pos[0] - 5, text_pos[1] - text_size[1] - 30),
                                        (text_pos[0] + text_size[0] + 50, text_pos[1] + 5),
                                        (0, 0, 0), -1)
                            
                            # Draw ratio and distance text
                            cv2.putText(frame, ratio_text, text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['track_point'], 2)
                            
                            cv2.putText(frame, dist_text, (text_pos[0], text_pos[1] - 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['track_point'], 2)
                        
                        # Highlight if shoulder drop detected
                        if shoulder_drop:
                            # Draw warning text for shoulder drop
                            warning_bg = frame.copy()
                            warning_text = "SHOULDER HUNCHING DETECTED"
                            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                            
                            w_center = w // 2
                            cv2.rectangle(warning_bg, 
                                         (w_center - text_size[0]//2 - 10, 120),
                                         (w_center + text_size[0]//2 + 10, 160),
                                         (0, 0, 255), -1)
                            cv2.addWeighted(warning_bg, 0.7, frame, 0.3, 0, frame)
                            
                            cv2.putText(frame, warning_text, 
                                       (w_center - text_size[0]//2, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Draw all keypoints and their connections
                frame = self.draw_keypoints_info(frame, keypoints, w, h)
        
        # Draw calibration overlay if in progress
        if self.calibration_in_progress:
            # Calculate remaining time
            elapsed_time = time.time() - self.calibration_start_time
            remaining_time = max(0, self.calibration_duration - elapsed_time)
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Display calibration instructions
            cv2.putText(frame, "CALIBRATION IN PROGRESS", (w//2 - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Please sit with UPRIGHT posture", 
                       (w//2 - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Keep your head straight and look at the camera", 
                       (w//2 - 230, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time remaining: {remaining_time:.1f} seconds", 
                       (w//2 - 150, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display posture status if calibrated
        if self.calibrated:
            # Select status color based on posture
            color = self.colors['upright'] if posture == "Upright" else self.colors['hunched']
            
            # Create semi-transparent background for status
            status_overlay = frame.copy()
            cv2.rectangle(status_overlay, (20, 20), (350, 100), (0, 0, 0), -1)
            cv2.addWeighted(status_overlay, 0.7, frame, 0.3, 0, frame)
            
            # Display posture status with larger text
            cv2.putText(frame, f"Posture: {posture}", (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if neck_angle is not None:
                cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}°", (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw analytics on frame
            frame = self.draw_analytics(frame, posture)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display baseline shoulder drop distance under FPS
        if self.calibrated and self.baseline_shoulder_drop_distance is not None:
            cv2.putText(frame, f"Baseline: {self.baseline_shoulder_drop_distance:.2f} px", 
                       (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display current drop ratio below baseline
            if drop_ratio > 0:
                # Choose color based on threshold
                ratio_color = (255, 255, 255)  # Default white
                if drop_ratio > self.shoulder_drop_threshold:
                    ratio_color = self.colors['hunched']  # Red if above threshold
                
                cv2.putText(frame, f"Drop Ratio: {drop_ratio:.2f}x", 
                           (w - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color, 2)
        
        # Update last update time for statistics
        self.last_update_time = time.time()
        
        return frame, posture, neck_angle, shoulder_drop
    
    def run(self):
        """Run the posture detector with webcam feed"""
        cap = cv2.VideoCapture(0)
        
        # Set higher resolution if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start calibration
        self.start_calibration()
        
        print("Press 'q' to quit, 'c' to recalibrate")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                # Process frame
                frame, posture, angle, shoulder_drop = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('YOLOv11 Posture Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.start_calibration()
                    
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()  # Clean up pygame resources

if __name__ == "__main__":
    detector = PostureDetectorYOLO()
    detector.run()
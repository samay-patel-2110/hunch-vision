"""
Emotion Detection Module

This module provides functions to detect emotions from facial landmarks using
the valence-arousal model. It can be imported into other projects or run as
a standalone script with webcam input.

The module is structured with four main classes:
- EyeStateDetector: Handles eye state detection and blink counting
- EmotionDetector: Handles emotion detection from facial landmarks
- HunchDetector: Handles hunch detection based on shoulder drop
- FaceAnalyzer: Main class that coordinates the other classes
- PerformanceMonitor: Tracks system performance metrics
"""

import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
import argparse
import psutil  # For CPU and memory usage tracking
from eye import EyeStateDetector
from emotion import EmotionDetector
from hunch import HunchDetector
import signal
import sys

# Global flag for termination
terminate_program = False

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    global terminate_program
    print('\nReceived termination signal. Cleaning up...')
    terminate_program = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

class FaceAnalyzer:
    """Main class that coordinates eye state detection and emotion detection."""
    
    def __init__(self, history_size=30):
        """Initialize the face analyzer.
        
        Args:
            history_size: Number of frames to keep in the history (default: 30)
        """
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe pose detection for shoulder tracking
        self.mp_pose = mp.solutions.pose
        
        # Initialize face mesh with appropriate settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize pose detector for shoulder landmarks
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize eye state detector
        self.eye_detector = EyeStateDetector(history_size)
        
        # Initialize emotion detector
        self.emotion_detector = EmotionDetector(history_size)
        
        # Initialize hunch detector
        self.hunch_detector = HunchDetector(history_size)
    
    def start_calibration(self):
        """Start the calibration process for eye state detection."""
        self.eye_detector.start_calibration()
        self.hunch_detector.start_calibration()
    
    def draw_landmarks(self, frame, face_landmarks, w, h):
        """Draw facial landmarks on the frame with color coding.
        
        Args:
            frame: The frame to draw on
            face_landmarks: MediaPipe face landmarks
            w: Frame width
            h: Frame height
        """
        for idx, landmark in enumerate(face_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Color based on facial regions
            if idx in range(0, 68):  # Jaw region
                color = (0, 255, 0)  # Green
            elif idx in range(68, 151):  # Eye region
                color = (255, 0, 0)  # Blue
            elif idx in range(151, 200):  # Eyebrow region
                color = (0, 0, 255)  # Red
            elif idx in range(200, 300):  # Nose region
                color = (255, 255, 0)  # Yellow
            else:  # Mouth region
                color = (255, 0, 255)  # Magenta
                
            cv2.circle(frame, (x, y), 1, color, -1)
    
    def draw_eyebrow_eye_lines(self, frame, face_landmarks, w, h, expressions):
        """Enhanced visualization of eyebrow and eye relationships for frowness detection."""
        if not expressions or not self.eye_detector.calibrated:
            return
            
        # Get landmark points
        landmarks = face_landmarks.landmark
        
        # Get binary frowning state (0 or 1) instead of continuous intensity
        is_frowning = expressions.get("is_frowning", False)
        raw_is_frowning = expressions.get("raw_is_frowning", False) 
        frowness_intensity = expressions.get("frowness_intensity", 0.0)  # Will now be 0.0 or 1.0
        frowness_confidence = expressions.get("frowness_confidence", 0.0)
        
        # Set color based on binary frowning state and confidence
        if is_frowning:
            # Color intensity based on confidence (more intense red with higher confidence)
            r = 255
            g = max(0, int(255 * (1.0 - frowness_confidence)))
            b = max(0, int(255 * (1.0 - frowness_confidence)))
            color = (b, g, r)
        else:
            # Color intensity based on confidence (more intense green with higher confidence)
            r = max(0, int(255 * (1.0 - frowness_confidence)))
            g = 255
            b = max(0, int(255 * (1.0 - frowness_confidence)))
            color = (b, g, r)
        
        # Create a color gradient from green (0) to red (1) based on intensity
        # b = 0
        # g = int(255 * (1 - frowness_intensity))
        # r = int(255 * frowness_intensity)
        # color = (b, g, r)
        
        # Draw multiple eyebrow-eye connections with labels showing weights
        detector = self.eye_detector
        
        # Create a semi-transparent overlay for better visualization
        overlay = frame.copy()
        
        # Draw left eyebrow-eye connections with point weights
        for i, eb_point in enumerate(detector.left_eyebrow_landmarks):
            if i < len(detector.left_eye_upper_landmarks):
                eye_point = detector.left_eye_upper_landmarks[i]
                
                # Get weight for this point pair (if available)
                weight = detector.eyebrow_point_weights[i] if i < len(detector.eyebrow_point_weights) else 0.0
                
                # Convert to screen coordinates
                eb_x = int(landmarks[eb_point].x * w)
                eb_y = int(landmarks[eb_point].y * h)
                eye_x = int(landmarks[eye_point].x * w)
                eye_y = int(landmarks[eye_point].y * h)
                
                # Draw line connecting eyebrow to eye
                cv2.line(frame, (eb_x, eb_y), (eye_x, eye_y), color, 2)
                
                # Draw points with larger radius for higher weights
                radius = int(5 * weight + 2)  # Scale radius by weight
                cv2.circle(frame, (eb_x, eb_y), radius, color, -1)
                cv2.circle(frame, (eye_x, eye_y), radius, color, -1)
                
                # Add point weight labels
                midpoint_x = (eb_x + eye_x) // 2
                midpoint_y = (eb_y + eye_y) // 2
                cv2.putText(frame, f"{weight:.2f}", (midpoint_x, midpoint_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add landmark IDs (small yellow text)
                cv2.putText(frame, f"{eb_point}", (eb_x+5, eb_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.putText(frame, f"{eye_point}", (eye_x+5, eye_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Draw right eyebrow-eye connections
        for i, eb_point in enumerate(detector.right_eyebrow_landmarks):
            if i < len(detector.right_eye_upper_landmarks):
                eye_point = detector.right_eye_upper_landmarks[i]
                
                # Get weight for this point pair (if available)
                weight = detector.eyebrow_point_weights[i] if i < len(detector.eyebrow_point_weights) else 0.0
                
                # Convert to screen coordinates
                eb_x = int(landmarks[eb_point].x * w)
                eb_y = int(landmarks[eb_point].y * h)
                eye_x = int(landmarks[eye_point].x * w)
                eye_y = int(landmarks[eye_point].y * h)
                
                # Draw line connecting eyebrow to eye
                cv2.line(frame, (eb_x, eb_y), (eye_x, eye_y), color, 2)
                
                # Draw points with larger radius for higher weights
                radius = int(5 * weight + 2)  # Scale radius by weight
                cv2.circle(frame, (eb_x, eb_y), radius, color, -1)
                cv2.circle(frame, (eye_x, eye_y), radius, color, -1)
                
                # Add point weight labels
                midpoint_x = (eb_x + eye_x) // 2
                midpoint_y = (eb_y + eye_y) // 2
                cv2.putText(frame, f"{weight:.2f}", (midpoint_x, midpoint_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add landmark IDs (small yellow text)
                cv2.putText(frame, f"{eb_point}", (eb_x+5, eb_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.putText(frame, f"{eye_point}", (eye_x+5, eye_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Create a detailed metrics panel for frowness calibration
        # Calculate panel position based on eye_plot position on right side
        # First find the eye plot position
        eye_plot = self.eye_detector.plot_eye_distances(frame.copy())
        eye_plot_h, eye_plot_w = eye_plot.shape[:2]
        eye_x_offset = w - eye_plot_w - 20
        eye_y_offset = 50  # Move down a bit to make room for eye state indicator
        
        # Position metrics panel beneath the eye plot
        panel_x = eye_x_offset
        panel_y = eye_y_offset + eye_plot_h + 20  # 20px gap after eye plot
        line_height = 20
        
        # Background for metrics panel
        metrics_panel_height = 240
        metrics_panel_width = eye_plot_w
        cv2.rectangle(frame, (panel_x-5, panel_y-25), (panel_x+metrics_panel_width+5, panel_y+metrics_panel_height), 
                     (20, 20, 20), -1)
        cv2.rectangle(frame, (panel_x-5, panel_y-25), (panel_x+metrics_panel_width+5, panel_y+metrics_panel_height), 
                     (200, 200, 200), 1)
        
        # Panel title
        cv2.putText(frame, "FROWNESS METRICS PANEL", (panel_x + 10, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel_y += line_height + 5
        
        # Get frowness metrics from expressions
        rel_left = expressions.get("relative_left_eyebrow_eye_distance", 0.0)
        rel_right = expressions.get("relative_right_eyebrow_eye_distance", 0.0)
        frowness = expressions.get("frowness", 0.0)
        frowness_detailed = expressions.get("frowness_detailed", 0.0)
        weighted_depression = expressions.get("weighted_eyebrow_depression", 0.0)
        
        # Get baseline values
        baseline_left = detector.baseline_left_eyebrow_eye_distance
        baseline_right = detector.baseline_right_eyebrow_eye_distance
        threshold = detector.eyebrow_frown_threshold
        
        # Left eye baseline and current values
        cv2.putText(frame, f"LEFT EYE FROWNESS:", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        panel_y += line_height
        
        cv2.putText(frame, f"  Baseline: {baseline_left:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        panel_y += line_height
        
        # Color based on proximity to threshold
        left_color = (0, 255, 0)  # Default green
        if rel_left <= threshold:  # Below threshold
            left_color = (0, 0, 255)  # Red
            
        cv2.putText(frame, f"  Current: {expressions.get('left_eyebrow_eye_distance', 0.0):.4f}", 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panel_y += line_height
        
        cv2.putText(frame, f"  Relative: {rel_left:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1)
        panel_y += line_height
        
        # Right eye baseline and current values
        cv2.putText(frame, f"RIGHT EYE FROWNESS:", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        panel_y += line_height
        
        cv2.putText(frame, f"  Baseline: {baseline_right:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        panel_y += line_height
        
        # Color based on proximity to threshold
        right_color = (0, 255, 0)  # Default green
        if rel_right <= threshold:  # Below threshold
            right_color = (0, 0, 255)  # Red
            
        cv2.putText(frame, f"  Current: {expressions.get('right_eyebrow_eye_distance', 0.0):.4f}", 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panel_y += line_height
        
        cv2.putText(frame, f"  Relative: {rel_right:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1)
        panel_y += line_height
        
        # Threshold information
        cv2.putText(frame, f"THRESHOLD: {threshold:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
        panel_y += line_height
        
        # Distance from threshold for left and right eyes
        left_margin = rel_left - threshold
        right_margin = rel_right - threshold
        
        left_margin_color = (0, 255, 0) if left_margin > 0 else (0, 0, 255)
        right_margin_color = (0, 255, 0) if right_margin > 0 else (0, 0, 255)
        
        cv2.putText(frame, f"  Left margin: {left_margin:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_margin_color, 1)
        panel_y += line_height
        
        cv2.putText(frame, f"  Right margin: {right_margin:.4f}", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_margin_color, 1)
        panel_y += line_height
        
        # Determine eye frowness state based on margins
        frowness_state = "Normal"
        frowness_color = (0, 255, 0)  # Default green
        
        if left_margin < 0 and right_margin < 0:
            frowness_state = "Frowning"
            frowness_color = (0, 0, 255)  # Red
        elif left_margin < 0:
            frowness_state = "Left Frowning" 
            frowness_color = (0, 0, 255)  # Red
        elif right_margin < 0:
            frowness_state = "Right Frowning"
            frowness_color = (0, 0, 255)  # Red
        elif expressions.get("eyebrows_raised", False):
            frowness_state = "Raised"
            frowness_color = (0, 255, 255)  # Yellow
            
        cv2.putText(frame, f"Eye State: {frowness_state}", (panel_x, panel_y+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, frowness_color, 1)
        panel_y += line_height
    
    def draw_mouth_lines(self, frame, face_landmarks, w, h, expressions):
        """Draw lines showing the mouth height and width.
        
        Args:
            frame: The frame to draw on
            face_landmarks: MediaPipe face landmarks
            w: Frame width
            h: Frame height
            expressions: Dictionary containing facial expression information
        """
        if not expressions or not self.eye_detector.calibrated:
            return
            
        # Get landmark points
        landmarks = face_landmarks.landmark
        
        # Mouth height: upper lip (13) to lower lip (14)
        upper_lip_x = int(landmarks[13].x * w)
        upper_lip_y = int(landmarks[13].y * h)
        lower_lip_x = int(landmarks[14].x * w)
        lower_lip_y = int(landmarks[14].y * h)
        
        # Mouth width: left corner (78) to right corner (308)
        left_corner_x = int(landmarks[78].x * w)
        left_corner_y = int(landmarks[78].y * h)
        right_corner_x = int(landmarks[308].x * w)
        right_corner_y = int(landmarks[308].y * h)
        
        # Determine color based on mouth state
        if expressions.get("lips_squeezed", False):
            color = (0, 0, 255)  # Red for squeezed lips
        elif expressions.get("smiling", False):
            color = (0, 255, 0)  # Green for smiling
        elif expressions.get("frowning", False):
            color = (255, 0, 0)  # Blue for frowning
        else:
            color = (255, 255, 255)  # White for neutral
            
        # Draw mouth height line
        cv2.line(frame, (upper_lip_x, upper_lip_y), (lower_lip_x, lower_lip_y), color, 2)
        
        # Draw mouth width line
        cv2.line(frame, (left_corner_x, left_corner_y), (right_corner_x, right_corner_y), color, 2)
        
        # Draw circles at endpoints
        cv2.circle(frame, (upper_lip_x, upper_lip_y), 3, color, -1)
        cv2.circle(frame, (lower_lip_x, lower_lip_y), 3, color, -1)
        cv2.circle(frame, (left_corner_x, left_corner_y), 3, color, -1)
        cv2.circle(frame, (right_corner_x, right_corner_y), 3, color, -1)
        
        # # Display mouth height value
        # mouth_height = expressions.get("relative_mouth_height", 0.0)
        # cv2.putText(frame, f"Mouth Height: {mouth_height:.2f}", (10, 180),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def process_frame(self, frame):
        """Process a single frame and return analysis results.
        
        Args:
            frame: BGR image/frame from a video source
            
        Returns:
            dict: A dictionary containing analysis results
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Process with MediaPipe Pose for shoulders
        pose_results = self.pose.process(rgb_frame)
        
        # Default return values
        valence, arousal = 0.0, 0.0
        emotion_name = "Unknown"
        face_success = False
        eye_state = "unknown"
        eye_plot = None
        expressions = None
        blink_detected = False
        posture_info = {"calibrated": False, "hunch_state": "Unknown"}
        
        # Process shoulder posture if pose landmarks detected
        if pose_results.pose_landmarks:
            posture_info = self.hunch_detector.process_posture(
                pose_results.pose_landmarks.landmark, 
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            )
            
            # Draw posture indicators
            frame = self.hunch_detector.draw_posture_indicators(frame)
            
            # Draw posture analytics
            frame = self.hunch_detector.draw_analytics(frame)
        
        # If face landmarks detected
        if face_results.multi_face_landmarks:
            face_success = True
            h, w, _ = frame.shape
            
            # Draw landmarks on the frame
            for face_landmarks in face_results.multi_face_landmarks:
                # Convert landmarks to numpy array
                landmarks = np.array([(landmark.x, landmark.y, landmark.z) 
                                    for landmark in face_landmarks.landmark])
                
                # Get face size for normalization
                face_size = (np.linalg.norm(landmarks[33] - landmarks[263]) + 
                           np.linalg.norm(landmarks[152] - landmarks[10])) / 2
                
                # Process eye measurements
                eye_info = self.eye_detector.process_eye_measurements(landmarks, face_size)
                eye_state = eye_info["eye_state"]
                expressions = eye_info["expressions"]
                blink_detected = eye_info["blink_detected"]
                
                # Process emotion
                emotion_info = self.emotion_detector.process_emotion(
                    landmarks, expressions)
                valence = emotion_info["valence"]
                arousal = emotion_info["arousal"]
                emotion_name = emotion_info["emotion"]
                
                # Draw face landmarks
                self.draw_landmarks(frame, face_landmarks, w, h)
                
                # Draw eyebrow to eye distance lines
                self.draw_eyebrow_eye_lines(frame, face_landmarks, w, h, expressions)
                
                # Draw mouth lines
                self.draw_mouth_lines(frame, face_landmarks, w, h, expressions)
                
                # Draw eye state if calibrated
                if self.eye_detector.calibrated:
                    # Color based on eye state
                    if eye_state == "wide_open":
                        color = (0, 255, 255)  # Yellow
                    elif eye_state == "squinting":
                        color = (255, 255, 0)  # Cyan
                    elif eye_state == "blinking":
                        color = (255, 0, 255)  # Magenta
                    elif eye_state == "closed":
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green
                    
                    # Draw eye state indicator
                    cv2.rectangle(frame, (w - 150, 10), (w - 10, 40), color, -1)
                    cv2.putText(frame, eye_state, (w - 140, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Create eye distance plot
            eye_plot = self.eye_detector.plot_eye_distances(frame)
            
            # Position for eye plot
            h, w, _ = frame.shape
            eye_plot_h, eye_plot_w, _ = eye_plot.shape
            eye_x_offset = w - eye_plot_w - 20
            eye_y_offset = 50  # Move down a bit to make room for eye state indicator
            
            # Calculate frowness metrics panel height
            metrics_panel_height = 240
            
            # We're removing the shoulder plot display, but will keep it in the return value
            # for any analysis that might need it
            
            # Draw border around eye plot
            cv2.rectangle(frame, 
                         (eye_x_offset - 5, eye_y_offset - 5), 
                         (eye_x_offset + eye_plot_w + 5, eye_y_offset + eye_plot_h + 5), 
                         (255, 255, 255), 2)
            
            # Add eye plot to the frame
            frame[eye_y_offset:eye_y_offset+eye_plot_h, eye_x_offset:eye_x_offset+eye_plot_w] = eye_plot
            
            # Add label above eye plot
            cv2.putText(frame, "EYE OPENNESS", (eye_x_offset + 10, eye_y_offset - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display facial expression information if calibrated
            if self.eye_detector.calibrated and expressions:
                # Display eyebrow and mouth information below blink count
                y_offset = 130  # Start below blink count
                
                # # Eyebrow information
                # if expressions["eyebrows_raised"]:
                #     cv2.putText(frame, "Eyebrows: RAISED", (10, y_offset),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # elif expressions["eyebrows_frowning"]:
                #     cv2.putText(frame, "Eyebrows: FROWNING", (10, y_offset),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # else:
                #     cv2.putText(frame, "Eyebrows: NORMAL", (10, y_offset),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Mouth information
                y_offset += 30
                if expressions["smiling"]:
                    cv2.putText(frame, "Mouth: SMILING", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif expressions["frowning"]:
                    cv2.putText(frame, "Mouth: FROWNING", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif expressions["lips_squeezed"]:
                    cv2.putText(frame, "Mouth: SQUEEZED", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Mouth: NEUTRAL", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display emotion state
                y_offset += 30
                if emotion_name == "Relaxed":
                    color = (0, 255, 0)  # Green
                else: 
                    color = (0, 0, 255)  # Red
                cv2.putText(frame, f"Emotion: {emotion_name}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Display posture information
                y_offset += 30
                if posture_info["calibrated"]:
                    hunch_state = posture_info["hunch_state"]
                    if "Good" in hunch_state:
                        color = (0, 255, 0)  # Green
                    elif "Slight" in hunch_state:
                        color = (0, 255, 255)  # Yellow
                    elif "Medium" in hunch_state:
                        color = (0, 165, 255)  # Orange
                    else:  # Severe Hunch
                        color = (0, 0, 255)  # Red
                    
                    cv2.putText(frame, f"Posture: {hunch_state}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Display hunching duration if applicable
                    if posture_info.get("is_hunched", False) and posture_info.get("hunch_duration", 0) > 0:
                        y_offset += 30
                        duration = posture_info["hunch_duration"]
                        cv2.putText(frame, f"Hunched for: {duration:.1f}s", (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display calibration text if calibration is in progress
        if self.eye_detector.calibration_in_progress or self.hunch_detector.calibration_in_progress:
            # Calculate remaining calibration time for eye calibration
            if self.eye_detector.calibration_in_progress:
                elapsed_time = time.time() - self.eye_detector.calibration_start_time
                remaining_time = max(0, self.eye_detector.calibration_duration - elapsed_time)
                calibration_text = f"EYE CALIBRATION: {remaining_time:.1f}s remaining"
            else:
                # For shoulder calibration, count frames
                remaining_frames = max(0, 30 - len(self.hunch_detector.calibration_samples))
                calibration_text = f"POSTURE CALIBRATION: {remaining_frames} frames remaining"
            
            # Create a semi-transparent overlay for the text
            overlay = frame.copy()
            h, w, _ = frame.shape
            
            # Draw a semi-transparent background for the text
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Display calibration text
            cv2.putText(frame, "CALIBRATION IN PROGRESS", (w//2 - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, calibration_text, (w//2 - 200, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Please sit with UPRIGHT posture and look at the camera", 
                       (w//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Keep your shoulders level and back straight", 
                       (w//2 - 250, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return {
            "frame": frame,
            "valence": valence,
            "arousal": arousal,
            "emotion": emotion_name,
            "face_success": face_success,
            "blink_count": self.eye_detector.blink_counter,
            "eye_state": eye_state,
            "eye_plot": eye_plot,
            "expressions": expressions,
            "posture_info": posture_info
        }
    
    def release(self):
        """Release resources used by the analyzer."""
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
        except Exception as e:
            print(f"Warning: Error closing face mesh: {e}")
            
        try:
            if hasattr(self, 'pose') and self.pose is not None:
                self.pose.close()
        except Exception as e:
            print(f"Warning: Error closing pose detector: {e}")

class PerformanceMonitor:
    """Class for monitoring system performance metrics."""
    
    def __init__(self, history_size=100):
        """Initialize the performance monitor.
        
        Args:
            history_size: Number of measurements to keep in history
        """
        # Initialize history tracking
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.fps_history = deque(maxlen=history_size)
        
        # Last measurement time
        self.last_measure_time = time.time()
        
        # Measurement interval (seconds)
        self.measure_interval = 0.5  # Measure every 0.5 seconds
        
        # Performance thresholds
        self.cpu_alert_threshold = 20.0  # Alert when CPU > 20%
        
        # Current values
        self.current_cpu = 0.0
        self.current_memory = 0.0
        self.current_fps = 0.0
        self.average_cpu = 0.0
        self.average_memory = 0.0
        self.average_fps = 0.0
        
        # System process
        self.process = psutil.Process()
        
        # Take initial measurements
        self._measure()
    
    def _measure(self):
        """Take performance measurements."""
        # CPU usage (percent)
        try:
            self.current_cpu = psutil.cpu_percent(interval=0)
            self.cpu_history.append(self.current_cpu)
            self.average_cpu = sum(self.cpu_history) / len(self.cpu_history)
        except Exception as e:
            print(f"Error measuring CPU: {e}")
        
        # Memory usage (percent)
        try:
            self.current_memory = psutil.virtual_memory().percent
            self.memory_history.append(self.current_memory)
            self.average_memory = sum(self.memory_history) / len(self.memory_history)
        except Exception as e:
            print(f"Error measuring memory: {e}")
        
        # Update last measure time
        self.last_measure_time = time.time()
    
    def update(self, current_fps):
        """Update performance metrics.
        
        Args:
            current_fps: Current FPS measurement
        """
        # Update FPS tracking
        self.current_fps = current_fps
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 0:
            self.average_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Measure system metrics at intervals
        current_time = time.time()
        if current_time - self.last_measure_time >= self.measure_interval:
            self._measure()
    
    def is_cpu_high(self):
        """Check if CPU usage is above the alert threshold.
        
        Returns:
            bool: True if CPU usage is high
        """
        return self.current_cpu > self.cpu_alert_threshold
    
    def draw_metrics(self, frame):
        """Draw performance metrics on the frame.
        
        Args:
            frame: BGR frame to draw on
            
        Returns:
            frame: Frame with metrics drawn
        """
        h, w, _ = frame.shape
        
        # Position for metrics (top-right corner)
        metrics_x = 10
        metrics_y = 70
        
        # Draw metrics with value-based colors, without panel background
        # CPU usage
        cpu_color = (0, 255, 0)  # Green by default
        if self.current_cpu > self.cpu_alert_threshold:
            cpu_color = (0, 0, 255)  # Red if high
        
        cv2.putText(frame, f"CPU: {self.current_cpu:.1f}%", 
                   (metrics_x, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, cpu_color, 1)
        
        # Memory usage
        memory_color = (0, 255, 0)  # Green by default
        if self.current_memory > 80:
            memory_color = (0, 0, 255)  # Red if high
        elif self.current_memory > 60:
            memory_color = (0, 165, 255)  # Orange if medium
            
        cv2.putText(frame, f"MEM: {self.current_memory:.1f}%", 
                   (metrics_x, metrics_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, memory_color, 1)
        
        
        fps_color = (0, 255, 0)  # Green by default
        if self.current_fps < 15:
            fps_color = (0, 0, 255)  # Red if low
        elif self.current_fps < 25:
            fps_color = (0, 165, 255)  # Orange if medium
            
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (metrics_x, metrics_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 1)
        
        # Draw alert if CPU usage is high
        if self.is_cpu_high():
            # Draw blinking alert if CPU is high (blink based on time)
            if int(time.time() * 2) % 2 == 0:  # Blink twice per second
                alert_text = "HIGH CPU USAGE!"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Position alert at top of screen
                alert_x = w // 2 - text_size[0] // 2
                alert_y = 30
                
                # Draw alert background
                cv2.rectangle(frame,
                             (alert_x - 10, alert_y - 20),
                             (alert_x + text_size[0] + 10, alert_y + 5),
                             (0, 0, 255), -1)
                
                # Draw alert text
                cv2.putText(frame, alert_text,
                           (alert_x, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def run_demo(source=0):
    """Run a demo of the face analyzer with webcam input.
    
    Args:
        source: Camera index or video file path (default: 0 for default webcam)
        window_size: Size of the display windows as (width, height) tuple
    """
    # Create face analyzer
    analyzer = FaceAnalyzer()
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # # Set window size
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
    
    print("Press 'q' to quit, 'r' to reset blink counter, 'c' to recalibrate")
    
    # Start calibration
    analyzer.start_calibration()
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.1  # Update FPS every 0.1 seconds for more responsive display
    frame_times = deque(maxlen=30)  # Store the last 30 frame times for rolling average
    
    try:
        while True:
            frame_start_time = time.time()
            
            # Check the global termination flag
            if terminate_program:
                print("Termination flag detected. Exiting main loop...")
                break
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Process frame
            results = analyzer.process_frame(frame)
            
            # Calculate frame processing time
            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            frame_times.append(frame_time)
            
            # Calculate and update FPS more frequently
            frame_count += 1
            if frame_end_time - start_time > fps_update_interval:
                # Calculate FPS based on rolling average of frame times
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    fps = 1.0 / max(0.001, avg_frame_time)  # Avoid division by zero
                frame_count = 0
                start_time = frame_end_time
            
            # Update performance monitor
            performance_monitor.update(fps)
            
            # Draw performance metrics
            results['frame'] = performance_monitor.draw_metrics(results['frame'])
            
            # Draw blink count below FPS
            cv2.putText(results['frame'], f"Blinks: {analyzer.eye_detector.blink_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display results
            cv2.imshow('Face Analysis', results['frame'])
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                analyzer.eye_detector.reset_blink_counter()
                print("Blink counter reset")
            elif key == ord('c'):
                analyzer.start_calibration()
                print("Recalibration started. Please maintain good posture during calibration...")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError in main loop: {e}")
    finally:
        # Cleanup
        print("Releasing resources...")
        if 'cap' in locals() and cap is not None:
            cap.release()
            print("Camera released")
        
        if 'analyzer' in locals() and analyzer is not None:
            analyzer.release()
            print("Analyzer resources released")
            
        cv2.destroyAllWindows()
        # Small delay to allow windows to be destroyed properly
        cv2.waitKey(1)
        print("All windows closed")
        print("Clean termination complete")

if __name__ == "__main__":
    # When run directly, process command line arguments and start the demo
    parser = argparse.ArgumentParser(description='Face Analysis with Emotion and Eye State Detection')
    parser.add_argument('--source', type=str, default='0', 
                        help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--width', type=int, default=1280, help='Window width')
    parser.add_argument('--height', type=int, default=720, help='Window height')
    
    args = parser.parse_args()
    
    try:
        # Convert source to int for webcam, keep as string for file path
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        
        # Run the demo
        run_demo(source=source)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError starting program: {e}")
    finally:
        # Final cleanup
        print("Program terminated") 
import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque

class EyeStateDetector:
    """Class for detecting eye states and blinks from facial landmarks."""
    
    def __init__(self, history_size=30):
        """Initialize the eye state detector.
        
        Args:
            history_size: Number of frames to keep in the eye history (default: 30)
        """
        # Initialize blink detection variables
        self.blink_counter = 0
        self.blink_threshold = 0.2  # Threshold for eye closure to be considered a blink
        self.min_blink_frames = 2  # Minimum frames eye must be closed to count as blink
        self.closed_frame_counter = 0
        
        # Initialize eye lid distance history for plotting
        self.left_eye_history = deque(maxlen=history_size)
        self.right_eye_history = deque(maxlen=history_size)
        self.start_time = 0
        
        # Blink detection parameters for graph-based approach
        self.blink_detected = False
        self.blink_window_size = 5 # Window size for detecting dips
        self.blink_dip_threshold = 0.1  # Minimum relative dip to consider as blink
        self.blink_min_duration = 1  # Minimum duration of dip in frames
        self.blink_cooldown = 5  # Frames to wait before detecting another blink
        self.blink_cooldown_counter = 0
        
        # Calibration variables
        self.calibrated = False
        self.baseline_eye_openness = 0.0
        self.baseline_std = 0.0
        self.calibration_samples = []
        self.calibration_duration = 5  # seconds
        self.calibration_start_time = 0
        self.calibration_in_progress = False
        
        # Additional calibration metrics
        self.baseline_eyebrow_distance = 0.0
        self.baseline_eyebrow_height = 0.0
        self.baseline_mouth_height = 0.0
        self.baseline_mouth_width = 0.0
        
        # Eye frowness detection
        self.baseline_left_eyebrow_eye_distance = 0.0
        self.baseline_right_eyebrow_eye_distance = 0.0
        self.eyebrow_eye_distance_history = deque(maxlen=history_size)
        self.eyebrow_raise_threshold = 1.1 # Eyebrows are considered raised when 10% above baseline
        self.eyebrow_frown_threshold = 0.95  # Adjust for more sensitivity (was 0.95)
        
        # Eye state detection thresholds
        self.wide_open_threshold = 1.2  # Eyes are considered wide open when 20% above baseline
        self.squint_threshold = 0.9  # Eyes are considered squinting when 20% below baseline
        self.blink_threshold_factor = 0.6  # Eyes are considered blinking when 40% below baseline
        
        # Facial expression detection thresholds
        self.smile_threshold = 1.2  # Mouth is considered smiling when 20% above baseline width
        self.frown_threshold = 0.8  # Mouth is considered circle when 20% below baseline width
        self.lips_squeezed_threshold = 0.5  # Lips are considered squeezed when height is 30% below baseline
        
        # Enhanced eyebrow/frowness detection with multiple landmark points
        # Eyebrow landmark points (inner to outer)
        self.left_eyebrow_landmarks = [70, 63, 105, 66, 107]
        self.right_eyebrow_landmarks = [300, 293, 334, 296, 336]
        
        # Eye upper lid landmarks (inner to outer)
        self.left_eye_upper_landmarks = [159, 158, 157, 173, 133]
        self.right_eye_upper_landmarks = [386, 385, 384, 398, 463]
        
        # Add more points for forehead tracking
        self.left_forehead_landmarks = [67, 109, 10]  # Points above left eyebrow
        self.right_forehead_landmarks = [297, 338, 152]  # Points above right eyebrow
        
        # Additional history trackers for new measurements
        self.eyebrow_curvature_history = deque(maxlen=history_size)
        self.forehead_wrinkle_history = deque(maxlen=history_size)
        
        # Additional baselines for enhanced frowness detection
        self.baseline_left_eyebrow_curvature = 0.0
        self.baseline_right_eyebrow_curvature = 0.0
        self.baseline_forehead_wrinkle = 0.0
        
        # Multi-point baseline storage
        self.baseline_left_eyebrow_eye_points = []
        self.baseline_right_eyebrow_eye_points = []
        
        # Add temporal smoothing for more stable frowning detection
        self.frowning_history = deque(maxlen=3)  # Store last 3 frames of frowning state
        self.frowning_confidence = 0.0  # Confidence level in current frowning state
        self.frowning_debounce_frames = 3  # Number of frames to confirm state change
        
        # Initialize audio for cue (matching hunch_mediapipe.py)
        try:
            import pygame
            pygame.mixer.init()
            self.audio_initialized = True
            try:
                self.cue_sound = pygame.mixer.Sound("cue.mp3")
                print("Successfully loaded audio cue")
            except Exception as e:
                print(f"Error loading audio cue: {e}")
                self.cue_sound = None
        except ImportError:
            print("Pygame not available, audio cues disabled")
            self.audio_initialized = False
            self.cue_sound = None
        
        # Add advanced frowness metrics
        self.frowness_weights = {
            "eyebrow_distance": 0.1,    # Weight for horizontal contraction
            "eyebrow_eye_distance": 0.7, # Weight for vertical distance (increased from 0.7)
            "eyebrow_curvature": 0.1,    # Weight for curved shape
            "forehead_wrinkle": 0.1      # Weight for forehead changes
        }
        
        # Add weights for multi-point eyebrow-eye measurements
        # Higher weights for inner points, lower for outer points
        # Increasing weights for inner points for more responsiveness
        self.eyebrow_point_weights = [0.45, 0.25, 0.15, 0.1, 0.05]  # Increased inner point weight
        
    def play_audio_cue(self):
        """Play audio cue when hunched posture is detected."""
        if self.cue_sound is None or not self.audio_initialized:
            return
        
        try:
            import threading
            # Play in a separate thread to avoid blocking
            def play_sound():
                self.cue_sound.play()
                
            threading.Thread(target=play_sound).start()
        except Exception as e:
            print(f"Error playing audio cue: {e}")
    
    def start_calibration(self):
        """Start the calibration process to establish a baseline for eye measurements."""
        self.calibration_in_progress = True
        self.calibration_samples = []
        self.calibration_start_time = time.time()
        self.calibrated = False
        print("Calibration started. Please gaze normally at the camera for 5 seconds...")
    
    def update_calibration(self, landmarks, face_size):
        """Update calibration with new measurements.
        
        Args:
            landmarks: Numpy array of facial landmarks
            face_size: Normalization factor based on face size
            
        Returns:
            bool: True if calibration is complete, False otherwise
        """
        if not self.calibration_in_progress:
            return False
            
        # Calculate eye openness
        left_eye_distance, right_eye_distance = self.calculate_eye_distances(landmarks, face_size)
        eye_openness = (left_eye_distance + right_eye_distance) / 2
        self.calibration_samples.append(eye_openness)
        
        # Calculate eyebrow distance (between inner corners of eyebrows)
        # Landmarks 105 and 334 are the inner corners of the eyebrows
        eyebrow_distance = np.linalg.norm(landmarks[105] - landmarks[334]) / face_size
        
        # Calculate eyebrow height (distance from eyebrow to eye)
        # Landmarks 105 and 145 for left eyebrow, 334 and 374 for right eyebrow
        left_eyebrow_height = np.linalg.norm(landmarks[105] - landmarks[145]) / face_size
        right_eyebrow_height = np.linalg.norm(landmarks[334] - landmarks[374]) / face_size
        eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
        
        # Calculate mouth height and width
        # Landmarks 13 and 14 for mouth height, 78 and 308 for mouth width
        mouth_height = np.linalg.norm(landmarks[13] - landmarks[14]) / face_size
        mouth_width = np.linalg.norm(landmarks[78] - landmarks[308]) / face_size
        
        # Calculate eyebrow to eye distance (frowness)
        # Left eye: distance from eyebrow (105) to upper eyelid (159)
        left_eyebrow_eye_distance = np.linalg.norm(landmarks[105] - landmarks[159]) / face_size
        # Right eye: distance from eyebrow (334) to upper eyelid (386)
        right_eyebrow_eye_distance = np.linalg.norm(landmarks[334] - landmarks[386]) / face_size
        
        # Check if calibration duration has elapsed
        elapsed_time = time.time() - self.calibration_start_time
        if elapsed_time >= self.calibration_duration:
            # Calculate baseline and standard deviation
            if len(self.calibration_samples) > 0:
                self.baseline_eye_openness = np.mean(self.calibration_samples)
                self.baseline_std = np.std(self.calibration_samples)
                
                # Set baseline for additional metrics
                self.baseline_eyebrow_distance = eyebrow_distance
                self.baseline_eyebrow_height = eyebrow_height
                self.baseline_mouth_height = mouth_height
                self.baseline_mouth_width = mouth_width
                
                # Set baseline for eyebrow to eye distance (frowness)
                self.baseline_left_eyebrow_eye_distance = left_eyebrow_eye_distance
                self.baseline_right_eyebrow_eye_distance = right_eyebrow_eye_distance

                
                self.calibrated = True
                self.start_time = time.time()
                self.calibration_in_progress = False
                print(f"Calibration complete. Baseline: {self.baseline_eye_openness:.4f}, Std: {self.baseline_std:.4f}")
                print(f"Eyebrow distance: {self.baseline_eyebrow_distance:.4f}, Eyebrow height: {self.baseline_eyebrow_height:.4f}")
                print(f"Mouth height: {self.baseline_mouth_height:.4f}, Mouth width: {self.baseline_mouth_width:.4f}")
                print(f"Eyebrow to eye distance (left): {self.baseline_left_eyebrow_eye_distance:.4f}")
                print(f"Eyebrow to eye distance (right): {self.baseline_right_eyebrow_eye_distance:.4f}")
                print(f"Multi-point eyebrow-eye distances with weights: {list(zip(self.eyebrow_point_weights, self.baseline_left_eyebrow_eye_points, self.baseline_right_eyebrow_eye_points))}")
                return True
            else:
                print("Calibration failed: No samples collected")
                self.calibration_in_progress = False
                return False
        
        return False
    
    def detect_eye_state(self, eye_openness):
        """Detect the current state of the eyes based on calibration baseline.
        
        Args:
            eye_openness: Current eye openness measurement
            
        Returns:
            str: Current eye state ("wide_open", "normal", "squinting", "blinking", "closed")
        """
        if not self.calibrated:
            return "unknown"
            
        # Calculate relative eye openness compared to baseline
        relative_openness = eye_openness / self.baseline_eye_openness
        
        # Determine eye state based on thresholds
        if relative_openness >= self.wide_open_threshold:
            return "wide_open"
        elif relative_openness <= self.blink_threshold_factor:
            return "blinking"
        elif relative_openness <= self.squint_threshold:
            return "squinting"
        elif relative_openness <= 0.1:  # Almost completely closed
            return "closed"
        else:
            return "normal"
    
    def detect_facial_expressions(self, landmarks, face_size):
        """Enhanced facial expression detection with improved frowness measurement."""
        if not self.calibrated:
            return {
                "eyebrows_raised": False,
                "eyebrows_frowning": False,
                "smiling": False,
                "frowning": False,
                "lips_squeezed": False,
                "eyebrow_distance": 0.0,
                "eyebrow_height": 0.0,
                "mouth_height": 0.0,
                "mouth_width": 0.0,
                "left_eyebrow_eye_distance": 0.0,
                "right_eyebrow_eye_distance": 0.0,
                "relative_left_eyebrow_eye_distance": 0.0,
                "relative_right_eyebrow_eye_distance": 0.0,
                "frowness": 0.0,
                "frowness_detailed": 0.0,
                "frowness_intensity": 0.0,
                "eyebrow_curvature": 0.0,
                "forehead_wrinkle": 0.0,
                "inner_eyebrow_depression": 0.0,
                "weighted_eyebrow_depression": 0.0,
                "frowness_confidence": 0.0,
                "raw_is_frowning": False
            }
        
        # Calculate eyebrow distance (between inner corners of eyebrows)
        eyebrow_distance = np.linalg.norm(landmarks[105] - landmarks[334]) / face_size
        
        # Calculate eyebrow height (distance from eyebrow to eye)
        left_eyebrow_height = np.linalg.norm(landmarks[105] - landmarks[145]) / face_size
        right_eyebrow_height = np.linalg.norm(landmarks[334] - landmarks[374]) / face_size
        eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
        
        # Calculate mouth height and width
        # Landmarks 13 and 14 for mouth height, 78 and 308 for mouth width
        mouth_height = np.linalg.norm(landmarks[13] - landmarks[14]) / face_size
        mouth_width = np.linalg.norm(landmarks[78] - landmarks[308]) / face_size
        
        # Calculate eyebrow to eye distance (frowness)
        # Left eye: distance from eyebrow (105) to upper eyelid (159)
        left_eyebrow_eye_distance = np.linalg.norm(landmarks[105] - landmarks[159]) / face_size
        # Right eye: distance from eyebrow (334) to upper eyelid (386)
        right_eyebrow_eye_distance = np.linalg.norm(landmarks[334] - landmarks[386]) / face_size
        
        # Calculate relative measurements
        relative_eyebrow_distance = eyebrow_distance / self.baseline_eyebrow_distance
        relative_eyebrow_height = eyebrow_height / self.baseline_eyebrow_height
        relative_mouth_height = mouth_height / self.baseline_mouth_height
        relative_mouth_width = mouth_width / self.baseline_mouth_width
        
        # Calculate relative eyebrow to eye distance (frowness)
        relative_left_eyebrow_eye_distance = left_eyebrow_eye_distance / self.baseline_left_eyebrow_eye_distance
        relative_right_eyebrow_eye_distance = right_eyebrow_eye_distance / self.baseline_right_eyebrow_eye_distance
            
        left_margin = relative_left_eyebrow_eye_distance - self.eyebrow_frown_threshold
        right_margin = relative_right_eyebrow_eye_distance - self.eyebrow_frown_threshold
        
        if left_margin < 0 and right_margin < 0:
            is_frowning = True
        elif left_margin < 0:
            is_frowning = True
        elif right_margin < 0:
            is_frowning = True
        else:
            is_frowning = False
        
        # Keep original frowness calculation for backward compatibility
        frowness = (relative_left_eyebrow_eye_distance + relative_right_eyebrow_eye_distance) / 2
        self.eyebrow_eye_distance_history.append(frowness)
        
        # Return enhanced dictionary with all metrics
        return {
            "eyebrows_raised": relative_eyebrow_height >= self.eyebrow_raise_threshold,
            "eyebrows_frowning": relative_eyebrow_distance <= self.eyebrow_frown_threshold,
            "smiling": relative_mouth_width >= self.smile_threshold,
            "frowning": relative_mouth_width <= self.frown_threshold,
            "lips_squeezed": relative_mouth_height <= self.lips_squeezed_threshold,
            "eyebrow_distance": eyebrow_distance,
            "eyebrow_height": eyebrow_height,
            "mouth_height": mouth_height,
            "mouth_width": mouth_width,
            "relative_eyebrow_distance": relative_eyebrow_distance,
            "relative_eyebrow_height": relative_eyebrow_height,
            "relative_mouth_height": relative_mouth_height,
            "relative_mouth_width": relative_mouth_width,
            "left_eyebrow_eye_distance": left_eyebrow_eye_distance,
            "right_eyebrow_eye_distance": right_eyebrow_eye_distance,
            "relative_left_eyebrow_eye_distance": relative_left_eyebrow_eye_distance,
            "relative_right_eyebrow_eye_distance": relative_right_eyebrow_eye_distance,
            "frowness": frowness,
            "frowness_confidence": self.frowning_confidence,
            "is_frowning": is_frowning
        }
    
    def detect_blink_from_graph(self, eye_openness):
        """Detect blinks by analyzing dips in the eye openness graph.
        
        Args:
            eye_openness: The current eye openness value
            
        Returns:
            bool: True if a blink was detected, False otherwise
        """
        # Add current eye openness to history
        self.left_eye_history.append(eye_openness)
        
        # If we're in cooldown period, decrement counter and return
        if self.blink_cooldown_counter > 0:
            self.blink_cooldown_counter -= 1
            return False
        
        # Need enough history to detect a dip
        if len(self.left_eye_history) < self.blink_window_size:
            return False
        
        # Check if we're in a dip (blink)
        current_value = eye_openness
        window_values = list(self.left_eye_history)[-self.blink_window_size:]
        
        # Calculate average of window excluding current value
        window_avg = sum(window_values[:-1]) / (len(window_values) - 1)
        
        # Calculate relative dip
        relative_dip = (window_avg - current_value) / max(0.01, window_avg)
        
        # Check if this is a significant dip
        if relative_dip > self.blink_dip_threshold:
            # Check if we've been in a dip for minimum duration
            if self.closed_frame_counter >= self.blink_min_duration:
                # This is a valid blink
                self.blink_counter += 1
                # self.play_audio_cue()
                self.blink_detected = True
                self.closed_frame_counter = 0
                self.blink_cooldown_counter = self.blink_cooldown
                return True
            else:
                # We're in a dip but not long enough yet
                self.closed_frame_counter += 1
                return False
        else:
            # Not in a dip
            self.closed_frame_counter = 0
            return False

    def detect_blink(self, eye_openness):
        """Detect if a blink has occurred based on eye openness.
        
        Args:
            eye_openness: The normalized eye openness value
            
        Returns:
            bool: True if a blink was detected, False otherwise
        """
        # Use the graph-based blink detection
        return self.detect_blink_from_graph(eye_openness)

    def calculate_eye_distances(self, landmarks, face_size):
        """Calculate the distance between upper and lower eyelids for both eyes.
        
        Args:
            landmarks: Numpy array of facial landmarks
            face_size: Normalization factor based on face size
            
        Returns:
            tuple: (left_eye_distance, right_eye_distance) normalized by face size
        """
        # Left eye landmarks (upper lid: 159, lower lid: 145)
        left_eye_distance = np.linalg.norm(landmarks[159] - landmarks[145]) / face_size
        
        # Right eye landmarks (upper lid: 386, lower lid: 374)
        right_eye_distance = np.linalg.norm(landmarks[386] - landmarks[374]) / face_size
        
        return left_eye_distance, right_eye_distance

    def plot_eye_distances(self, frame, width=400, height=200):
        """Enhanced plot with additional frowness metrics visualization."""
        # Create a blank image for the plot
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        plot_img[:] = (30, 30, 30)  # Dark gray background
        
        # Draw grid lines
        for i in range(0, height, height//4):
            cv2.line(plot_img, (0, i), (width, i), (50, 50, 50), 1)
        
        # Draw left eye distance (green)
        if len(self.left_eye_history) > 1:
            points = []
            for i, dist in enumerate(self.left_eye_history):
                x = int(i * width / max(1, len(self.left_eye_history) - 1))
                y = int(height - dist * height * 5)  # Scale for visibility
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(plot_img, points[i], points[i+1], (0, 255, 0), 2)
            
            # Highlight blinks with red circles
            if self.blink_detected and len(points) > 0:
                # Draw a red circle at the last point to indicate a blink
                cv2.circle(plot_img, points[-1], 5, (0, 0, 255), -1)
                self.blink_detected = False  # Reset the flag
        
        # Draw right eye distance (blue)
        if len(self.right_eye_history) > 1:
            points = []
            for i, dist in enumerate(self.right_eye_history):
                x = int(i * width / max(1, len(self.right_eye_history) - 1))
                y = int(height - dist * height * 5)  # Scale for visibility
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(plot_img, points[i], points[i+1], (255, 0, 0), 2)
        
        # Draw enhanced frowness metrics (eyebrow curvature) in magenta
        if len(self.eyebrow_curvature_history) > 1:
            points = []
            for i, dist in enumerate(self.eyebrow_curvature_history):
                x = int(i * width / max(1, len(self.eyebrow_curvature_history) - 1))
                y = int(height - dist * height * 5)  # Scale for visibility
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(plot_img, points[i], points[i+1], (255, 0, 255), 2)
        
        # Draw forehead wrinkle in orange
        if len(self.forehead_wrinkle_history) > 1:
            points = []
            for i, dist in enumerate(self.forehead_wrinkle_history):
                x = int(i * width / max(1, len(self.forehead_wrinkle_history) - 1))
                y = int(height - dist * height * 5)  # Scale for visibility
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(plot_img, points[i], points[i+1], (0, 165, 255), 2)
        
        # Draw baseline if calibrated
        if self.calibrated:
            baseline_y = int(height - self.baseline_eye_openness * height * 5)
            cv2.line(plot_img, (0, baseline_y), (width, baseline_y), (255, 255, 255), 1)
            
            # Draw threshold lines
            wide_open_y = int(height - (self.baseline_eye_openness * self.wide_open_threshold) * height * 5)
            squint_y = int(height - (self.baseline_eye_openness * self.squint_threshold) * height * 5)
            blink_y = int(height - (self.baseline_eye_openness * self.blink_threshold_factor) * height * 5)
            
            cv2.line(plot_img, (0, wide_open_y), (width, wide_open_y), (0, 255, 255), 1)
            cv2.line(plot_img, (0, squint_y), (width, squint_y), (255, 255, 0), 1)
            cv2.line(plot_img, (0, blink_y), (width, blink_y), (255, 0, 255), 1)
        
        return plot_img
    
    def process_eye_measurements(self, landmarks, face_size):
        """Process eye measurements from facial landmarks.
        
        Args:
            landmarks: Numpy array of facial landmarks
            face_size: Normalization factor based on face size
            
        Returns:
            dict: Dictionary containing eye state information
        """
        # Calculate eye distances
        left_eye_distance, right_eye_distance = self.calculate_eye_distances(landmarks, face_size)
        
        # Add to history
        self.right_eye_history.append(right_eye_distance)
        
        # Calculate average eye openness for blink detection
        eye_openness = (left_eye_distance + right_eye_distance) / 2
        
        # Update calibration if in progress
        if self.calibration_in_progress:
            self.update_calibration(landmarks, face_size)
        
        # Detect blink using graph-based approach
        blink_detected = self.detect_blink(eye_openness)
        
        # Determine eye state based on calibration
        eye_state = self.detect_eye_state(eye_openness)
        
        # Detect facial expressions
        expressions = self.detect_facial_expressions(landmarks, face_size)
        
        return {
            "left_eye_distance": left_eye_distance,
            "right_eye_distance": right_eye_distance,
            "eye_openness": eye_openness,
            "blink_detected": blink_detected,
            "eye_state": eye_state,
            "blink_counter": self.blink_counter,
            "expressions": expressions
        }
    
    def reset_blink_counter(self):
        """Reset the blink counter to zero."""
        self.blink_counter = 0
        print("Blink counter reset")

if __name__ == "__main__":
    app = EyeStateDetector()

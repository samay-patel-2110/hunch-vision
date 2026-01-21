import numpy as np
import cv2
import time
from collections import deque   

class HunchDetector:
    """Class for detecting hunched posture based on shoulder drop."""
    
    def __init__(self, history_size=30):
        """Initialize the hunch detector.
        
        Args:
            history_size: Number of frames to keep in the history (default: 30)
        """
        # Shoulder landmark indices
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        
        # Initialize tracking variables
        self.mid_shoulder_history = deque(maxlen=history_size)
        self.reference_point_history = deque(maxlen=history_size)
        self.shoulder_drop_history = deque(maxlen=history_size)
        
        # Optical flow tracking
        self.prev_frame = None
        self.prev_reference_point = None
        self.frame_count = 0  # For frame skipping
        self.flow_calculation_interval = 2  # Calculate flow every n frames
        self.last_flow = None  # Store last calculated flow
        
        # Performance optimization
        self.flow_resize_factor = 0.5  # Resize factor for optical flow calculation
        
        # Calibration variables
        self.calibrated = False
        self.baseline_shoulder_drop = 0.0
        self.calibration_samples = []
        self.calibration_in_progress = False
        
        # Thresholds for hunch detection (aligning with hunch_mediapipe.py)
        self.shoulder_drop_threshold = 1.08  # 8% increase in distance (matching hunch_mediapipe.py)
        self.slight_hunch_threshold = 1.08  # For slight hunching
        self.medium_hunch_threshold = 1.15  # For medium hunching
        self.severe_hunch_threshold = 1.25  # For severe hunching
        
        # Last detected values
        self.mid_shoulder = None
        self.reference_point = None
        self.current_shoulder_drop = 0.0
        self.current_hunch_state = "Unknown"
        
        # Analytics tracking (added to match hunch_mediapipe.py)
        self.posture_stats = {
            "upright_time": 0,
            "hunched_time": 0,
            "total_time": 0
        }
        self.start_hunch_time = None
        self.hunch_duration = 0
        self.last_update_time = time.time()
        
        # # Initialize audio for cue
        # try:
        #     import pygame
        #     pygame.mixer.init()
        #     self.audio_initialized = True
        #     try:
        #         self.cue_sound = pygame.mixer.Sound("cue.mp3")
        #         print("Successfully loaded audio cue")
        #     except Exception as e:
        #         print(f"Error loading audio cue: {e}")
        #         self.cue_sound = None
        # except ImportError:
        #     print("Pygame not available, audio cues disabled")
        #     self.audio_initialized = False
        #     self.cue_sound = None

        # Audio control variables
        self.last_audio_time = 0
        self.audio_cooldown = 5  # seconds between audio cues
        self.was_hunched = False  # Track previous state
        
        # Colors for visualization (matching hunch_mediapipe.py)
        self.colors = {
            'upright': (0, 255, 0),  # Green
            'hunched': (0, 0, 255),  # Red
            'midpoint': (255, 255, 0),  # Yellow
            'angle': (255, 0, 255),  # Magenta
            'reference': (0, 255, 255),  # Cyan
            'track_point': (0, 140, 255)  # Orange for tracking point
        }
    
    def start_calibration(self):
        """Start the calibration process to establish baseline shoulder position."""
        self.calibration_in_progress = True
        self.calibration_samples = []
        self.calibrated = False
        print("Shoulder posture calibration started...")
        
        # Reset analytics when recalibrating
        self.posture_stats = {
            "upright_time": 0,
            "hunched_time": 0,
            "total_time": 0
        }
        self.hunch_duration = 0
        self.start_hunch_time = None
        self.last_update_time = time.time()
    
    def calculate_reference_point(self, landmarks):
        """Calculate the mid-shoulder and reference point below it.
        
        Args:
            landmarks: Array of body landmarks
            
        Returns:
            tuple: (mid_shoulder, reference_point) as numpy arrays
        """
        # Get shoulder landmarks
        left_shoulder = np.array([landmarks[self.LEFT_SHOULDER].x, 
                                 landmarks[self.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.RIGHT_SHOULDER].x, 
                                  landmarks[self.RIGHT_SHOULDER].y])
        
        # Calculate mid-shoulder point
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        
        # Calculate shoulder width
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Calculate reference point below mid-shoulder (1/4 of shoulder width)
        # Use the vector pointing downward (y increases as we go down)
        reference_point = mid_shoulder + np.array([0, shoulder_width * 0.25])
        
        return mid_shoulder, reference_point
    
    def update_calibration(self, landmarks):
        """Update calibration with new measurements.
        
        Args:
            landmarks: Array of body landmarks
            
        Returns:
            bool: True if calibration is complete, False otherwise
        """
        if not self.calibration_in_progress:
            return False
        
        # Calculate mid-shoulder and reference point
        mid_shoulder, reference_point = self.calculate_reference_point(landmarks)
        
        # Calculate distance between mid-shoulder and reference point
        shoulder_drop = np.linalg.norm(reference_point - mid_shoulder)
        
        # Add to calibration samples
        self.calibration_samples.append(shoulder_drop)
        
        # Check if we have enough samples (30 frames ≈ 1 second at 30fps)
        if len(self.calibration_samples) >= 30:
            # Calculate baseline as average of samples
            self.baseline_shoulder_drop = np.mean(self.calibration_samples)
            self.calibrated = True
            self.calibration_in_progress = False
            print(f"Shoulder posture calibration complete. Baseline: {self.baseline_shoulder_drop:.4f}")
            return True
        
        return False
    
    def track_with_optical_flow(self, curr_frame, ref_point):
        """Track reference point using optical flow.
        
        Args:
            curr_frame: Current grayscale frame
            ref_point: Reference point to track
            
        Returns:
            numpy.ndarray: Updated reference point location
        """
        if self.prev_frame is None or self.prev_reference_point is None:
            self.prev_frame = curr_frame.copy()
            self.prev_reference_point = ref_point
            return ref_point
        
        # Create DISOpticalFlow object if not already created (using fast preset)
        if not hasattr(self, 'dis_flow'):
            self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            
        # Frame skipping for performance
        self.frame_count += 1
        calculate_flow = (self.frame_count % self.flow_calculation_interval == 0)
        
        if calculate_flow:
            # Resize frames for faster processing
            h, w = curr_frame.shape[:2]
            small_h, small_w = int(h * self.flow_resize_factor), int(w * self.flow_resize_factor)
            
            # Skip if frames are too small
            if small_h <= 0 or small_w <= 0:
                return ref_point
                
            small_prev = cv2.resize(self.prev_frame, (small_w, small_h))
            small_curr = cv2.resize(curr_frame, (small_w, small_h))
            
            # Calculate flow on smaller frames
            self.last_flow = self.dis_flow.calc(small_prev, small_curr, None)
            
            # Scale flow back to original size
            self.last_flow = cv2.resize(self.last_flow, (w, h))
            self.last_flow[:,:,0] *= (w / small_w)
            self.last_flow[:,:,1] *= (h / small_h)
        
        # If we don't have flow yet, return unchanged point
        if self.last_flow is None:
            return ref_point
            
        # Get flow at reference point location
        h, w = self.last_flow.shape[:2]
        px, py = int(self.prev_reference_point[0] * w), int(self.prev_reference_point[1] * h)
        
        # Ensure points are within image bounds
        px = max(0, min(px, w-1))
        py = max(0, min(py, h-1))
        
        # Get flow vector at the reference point
        flow_x = self.last_flow[py, px, 0]
        flow_y = self.last_flow[py, px, 1]
        
        # Calculate new position
        new_x = self.prev_reference_point[0] + flow_x/w  # Normalize by width to keep in 0-1 range
        new_y = self.prev_reference_point[1] + flow_y/h  # Normalize by height to keep in 0-1 range
        
        # Keep coordinates in valid range (0-1)
        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))
        
        tracked_point = np.array([new_x, new_y])
        
        # Update previous frame and point for next iteration
        if calculate_flow:
            self.prev_frame = curr_frame.copy()
        self.prev_reference_point = tracked_point
        
        return tracked_point
    
    # def play_audio_cue(self):
    #     """Play audio cue when hunched posture is detected."""
    #     if self.cue_sound is None or not self.audio_initialized:
    #         return
        
    #     try:
    #         import threading
    #         # Play in a separate thread to avoid blocking
    #         def play_sound():
    #             self.cue_sound.play()
                
    #         threading.Thread(target=play_sound).start()
    #     except Exception as e:
    #         print(f"Error playing audio cue: {e}")
    
    def process_posture(self, landmarks, frame=None):
        """Process posture based on shoulder landmarks.
        
        Args:
            landmarks: Array of body landmarks
            frame: Optional frame for optical flow (grayscale)
            
        Returns:
            dict: A dictionary containing hunch detection information
        """
        # Update calibration if in progress
        if self.calibration_in_progress:
            self.update_calibration(landmarks)
            return {
                "calibrated": False,
                "hunch_state": "Calibrating",
                "relative_drop": 0.0,
                "mid_shoulder": None,
                "reference_point": None
            }
        
        # Calculate mid-shoulder and reference point
        mid_shoulder, reference_point = self.calculate_reference_point(landmarks)
        
        # Track reference point with optical flow if frame is provided
        if frame is not None:
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            reference_point = self.track_with_optical_flow(gray_frame, reference_point)
        
        # Calculate distance between mid-shoulder and reference point
        shoulder_drop = np.linalg.norm(reference_point - mid_shoulder)
        
        # Add to history
        self.mid_shoulder_history.append(mid_shoulder)
        self.reference_point_history.append(reference_point)
        self.shoulder_drop_history.append(shoulder_drop)
        
        # Calculate relative drop compared to baseline
        relative_drop = 0.0
        hunch_state = "Unknown"
        is_hunched = False
        
        if self.calibrated and self.baseline_shoulder_drop > 0:
            relative_drop = shoulder_drop / self.baseline_shoulder_drop
            
            # Determine hunch state based on thresholds
            if relative_drop >= self.severe_hunch_threshold:
                hunch_state = "Severe Hunch"
                is_hunched = True
            elif relative_drop >= self.medium_hunch_threshold:
                hunch_state = "Medium Hunch"
                is_hunched = True
            elif relative_drop >= self.slight_hunch_threshold:
                hunch_state = "Slight Hunch"
                is_hunched = True
            else:
                hunch_state = "Good Posture"
                is_hunched = False
            
            # Update analytics
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.posture_stats["total_time"] += dt
            
            if is_hunched:
                # Track continuous hunching
                if self.start_hunch_time is None:
                    self.start_hunch_time = current_time
                    
                    # # Play audio cue when transitioning to hunched state
                    # if not self.was_hunched and (current_time - self.last_audio_time > self.audio_cooldown):
                    #     self.play_audio_cue()
                    #     self.last_audio_time = current_time
                
                self.hunch_duration = current_time - self.start_hunch_time
                self.posture_stats["hunched_time"] += dt
            else:
                # Reset hunch time if posture corrected
                if self.start_hunch_time is not None:
                    self.hunch_duration = 0
                    self.start_hunch_time = None
                
                self.posture_stats["upright_time"] += dt
            
            # Update previous state
            self.was_hunched = is_hunched
            self.last_update_time = current_time
        
        # Update current values
        self.mid_shoulder = mid_shoulder
        self.reference_point = reference_point
        self.current_shoulder_drop = shoulder_drop
        self.current_hunch_state = hunch_state
        
        return {
            "calibrated": self.calibrated,
            "hunch_state": hunch_state,
            "relative_drop": relative_drop,
            "mid_shoulder": mid_shoulder,
            "reference_point": reference_point,
            "is_hunched": is_hunched,
            "hunch_duration": self.hunch_duration,
            "posture_stats": self.posture_stats
        }
    
    def draw_posture_indicators(self, frame):
        """Draw posture indicators on the frame.
        
        Args:
            frame: BGR frame to draw on
            
        Returns:
            frame: Frame with posture indicators drawn
        """
        if not self.calibrated or self.mid_shoulder is None or self.reference_point is None:
            return frame
        
        h, w, _ = frame.shape
        
        # Convert normalized coordinates to pixel coordinates
        mid_shoulder_px = (int(self.mid_shoulder[0] * w), int(self.mid_shoulder[1] * h))
        reference_point_px = (int(self.reference_point[0] * w), int(self.reference_point[1] * h))
        
        # Determine color based on hunch state
        if "Good" in self.current_hunch_state:
            color = self.colors['upright']  # Green
        elif "Slight" in self.current_hunch_state:
            color = (0, 255, 255)  # Yellow
        elif "Medium" in self.current_hunch_state:
            color = (0, 165, 255)  # Orange
        else:  # Severe Hunch
            color = self.colors['hunched']  # Red
        
        # Draw mid-shoulder point with black outline
        cv2.circle(frame, mid_shoulder_px, 7, (0, 0, 0), -1)  # Black outline
        cv2.circle(frame, mid_shoulder_px, 5, self.colors['midpoint'], -1)  # Yellow fill
        
        # Draw reference point with black outline
        cv2.circle(frame, reference_point_px, 7, (0, 0, 0), -1)  # Black outline
        cv2.circle(frame, reference_point_px, 5, self.colors['track_point'], -1)  # Orange fill
        
        # Draw line connecting mid-shoulder to reference point
        cv2.line(frame, mid_shoulder_px, reference_point_px, color, 2)
        
        # Draw text with hunch state and relative drop
        text = f"{self.current_hunch_state} ({self.current_shoulder_drop:.2f})"
        
        # Add background for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            frame, 
            (mid_shoulder_px[0] - 5, mid_shoulder_px[1] - 35),
            (mid_shoulder_px[0] + text_size[0] + 10, mid_shoulder_px[1] - 5),
            (0, 0, 0), 
            -1
        )
        
        cv2.putText(frame, text, (mid_shoulder_px[0], mid_shoulder_px[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # If hunched for more than 5 seconds, show duration warning
        if self.hunch_duration > 5 and "Good" not in self.current_hunch_state:
            warning_text = f"Hunched for {self.hunch_duration:.1f}s"
            
            # Add background for warning
            warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame, 
                (mid_shoulder_px[0] - 5, mid_shoulder_px[1] + 5),
                (mid_shoulder_px[0] + warning_size[0] + 10, mid_shoulder_px[1] + 35),
                (0, 0, 0), 
                -1
            )
            
            cv2.putText(frame, warning_text, (mid_shoulder_px[0], mid_shoulder_px[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['hunched'], 2)
        
        return frame
    
    def draw_analytics(self, frame):
        """Draw analytics information on the frame.
        
        Args:
            frame: BGR frame to draw on
            
        Returns:
            frame: Frame with analytics drawn
        """
        if not self.calibrated:
            return frame
            
        h, w, _ = frame.shape
        
        # Calculate percentages
        total = max(1, self.posture_stats["total_time"])
        upright_pct = (self.posture_stats["upright_time"] / total) * 100
        hunched_pct = (self.posture_stats["hunched_time"] / total) * 100
        
        # Create analytics panel at bottom
        panel_height = 80
        panel_y = h - panel_height
        
        # Draw semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw title
        cv2.putText(frame, "POSTURE ANALYTICS", (20, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw time bars
        bar_width = w - 40
        bar_height = 20
        bar_x = 20
        bar_y = panel_y + 30
        
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
        if self.hunch_duration > 0:
            cv2.putText(frame, f"Current Hunched Duration: {self.hunch_duration:.1f}s", 
                       (bar_x, bar_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.colors['hunched'], 2)
            
            if self.hunch_duration > 30:  # Warning after 30 seconds of hunching
                cv2.putText(frame, "WARNING: Consider adjusting your posture!", 
                           (bar_x, bar_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 0, 255), 2)
        
        return frame
    
    def plot_shoulder_drop(self, width=400, height=100):
        """Create a plot of shoulder drop history.
        
        Args:
            width: Width of plot image
            height: Height of plot image
            
        Returns:
            numpy.ndarray: Plot image
        """
        # Create a blank image for the plot
        plot_img = np.zeros((height, width, 3), dtype=np.uint8)
        plot_img[:] = (30, 30, 30)  # Dark gray background
        
        # Draw grid lines
        for i in range(0, height, height//4):
            cv2.line(plot_img, (0, i), (width, i), (50, 50, 50), 1)
        
        # Draw drop history
        if len(self.shoulder_drop_history) > 1 and self.calibrated:
            points = []
            for i, drop in enumerate(self.shoulder_drop_history):
                x = int(i * width / max(1, len(self.shoulder_drop_history) - 1))
                y = int(height - (drop / self.baseline_shoulder_drop) * height / 3)
                y = max(0, min(height-1, y))  # Clamp to plot bounds
                points.append((x, y))
            
            # Draw lines connecting points
            for i in range(len(points) - 1):
                # Color based on hunch severity
                relative_drop = self.shoulder_drop_history[i] / self.baseline_shoulder_drop
                if relative_drop >= self.severe_hunch_threshold:
                    color = self.colors['hunched']  # Red
                elif relative_drop >= self.medium_hunch_threshold:
                    color = (0, 165, 255)  # Orange
                elif relative_drop >= self.slight_hunch_threshold:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = self.colors['upright']  # Green
                
                cv2.line(plot_img, points[i], points[i+1], color, 2)
        
        # Draw threshold lines if calibrated
        if self.calibrated:
            # Draw baseline
            baseline_y = int(height - height / 3)
            cv2.line(plot_img, (0, baseline_y), (width, baseline_y), (255, 255, 255), 1)
            
            # Draw threshold lines
            slight_y = int(height - (self.slight_hunch_threshold) * height / 3)
            medium_y = int(height - (self.medium_hunch_threshold) * height / 3)
            severe_y = int(height - (self.severe_hunch_threshold) * height / 3)
            
            cv2.line(plot_img, (0, slight_y), (width, slight_y), (0, 255, 255), 1)
            cv2.line(plot_img, (0, medium_y), (width, medium_y), (0, 165, 255), 1)
            cv2.line(plot_img, (0, severe_y), (width, severe_y), (0, 0, 255), 1)
            
            # Draw threshold labels
            cv2.putText(plot_img, f"{self.slight_hunch_threshold:.2f}x", (width - 45, slight_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(plot_img, f"{self.medium_hunch_threshold:.2f}x", (width - 45, medium_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.putText(plot_img, f"{self.severe_hunch_threshold:.2f}x", (width - 45, severe_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return plot_img

if __name__ == "__main__":
    app = HunchDetector()


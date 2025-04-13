import time
from collections import deque

class EmotionDetector:
    """Class for detecting emotions based on facial expressions and blink rate."""
    
    def __init__(self, history_size=30):
        """Initialize the emotion detector.
        
        Args:
            history_size: Number of frames to keep in the history (default: 30)
        """
        # Define emotions and their visualization colors
        self.emotions = {
            "Relaxed": (0, 255, 0),  # Green
            "Happy": (255, 255, 0),  # Yellow
            "Angry": (0, 0, 255),    # Red
            "Stressed": (0, 0, 255)  # Red (kept for backward compatibility)
        }
        
        # Initialize blink tracking
        self.blink_times = deque(maxlen=60)  # Track last 60 blinks
        self.blink_rate = 0.0  # Blinks per minute
        
        # Initialize emotion history
        self.emotion_history = deque(maxlen=history_size)
        self.valence_history = deque(maxlen=history_size)
        self.arousal_history = deque(maxlen=history_size)
        
        # Blink rate thresholds
        self.low_blink_threshold = 10  # Less than 10 blinks per minute is low
        self.high_blink_threshold = 30  # More than 30 blinks per minute is high
    
    def update_blink_rate(self, blink_detected):
        """Update the blink rate based on detected blinks.
        
        Args:
            blink_detected: Boolean indicating if a blink was detected
        """
        if blink_detected:
            current_time = time.time()
            self.blink_times.append(current_time)
            
            # Calculate blink rate (blinks per minute)
            if len(self.blink_times) > 1:
                time_span = current_time - self.blink_times[0]
                if time_span > 0:
                    self.blink_rate = (len(self.blink_times) - 1) * 60 / time_span
    
    def get_blink_rate_status(self):
        """Get the status of the blink rate (Low, Normal, or High).
        
        Returns:
            str: The status of the blink rate
        """
        if self.blink_rate < self.low_blink_threshold:
            return "Low"
        elif self.blink_rate > self.high_blink_threshold:
            return "High"
        else:
            return "Normal"
    
    def calculate_emotion(self, expressions, blink_detected):
        """Calculate emotion based on facial expressions and blink rate.
        
        Args:
            expressions: Dictionary containing facial expression information
            blink_detected: Boolean indicating if a blink was detected
            
        Returns:
            tuple: (valence, arousal, emotion_name) values for the emotion
        """
        # Update blink rate
        self.update_blink_rate(blink_detected)
        
        # Get key expressions for emotion detection
        is_frowning = expressions.get("eyebrows_frowning", False) or expressions.get("frowning", False)
        is_smiling = expressions.get("smiling", False)
        
        # Determine emotion based on the specified rules
        if is_frowning and not is_smiling:
            emotion_name = "Angry"
            valence = -0.8
            arousal = 0.8
        elif not is_frowning and is_smiling:
            emotion_name = "Happy"
            valence = 0.8
            arousal = 0.5
        elif not is_frowning:
            emotion_name = "Relaxed"
            valence = 0.5
            arousal = 0.0
        else:
            # Fallback case - shouldn't really happen given the rules
            emotion_name = "Stressed"
            valence = -0.5
            arousal = 0.5
        
        # Update history
        self.valence_history.append(valence)
        self.arousal_history.append(arousal)
        self.emotion_history.append(emotion_name)
        
        return valence, arousal, emotion_name
    
    def get_emotion_name(self, valence, arousal):
        """Get the emotion name based on valence and arousal values.
        
        Args:
            valence: The valence value (-1 to 1)
            arousal: The arousal value (-1 to 1)
            
        Returns:
            str: The name of the emotion
        """
        # This method is now overridden by the calculate_emotion logic
        # but kept for backward compatibility
        if valence > 0:
            return "Relaxed"
        else:
            return "Stressed"
    
    def process_emotion(self, landmarks, expressions, blink_detected):
        """Process emotion based on facial landmarks and expressions.
        
        Args:
            landmarks: numpy array of facial landmarks
            expressions: Dictionary containing facial expression information
            blink_detected: Boolean indicating if a blink was detected
            
        Returns:
            dict: A dictionary containing:
                - valence: The calculated valence value (-1 to 1)
                - arousal: The calculated arousal value (-1 to 1)
                - emotion: The detected emotion name
                - blink_rate_status: The status of the blink rate (Low, Normal, High)
        """
        # Calculate emotion
        valence, arousal, emotion_name = self.calculate_emotion(expressions, blink_detected)
        
        # Get blink rate status
        blink_rate_status = self.get_blink_rate_status()
        
        return {
            "valence": valence,
            "arousal": arousal,
            "emotion": emotion_name,
            "blink_rate_status": blink_rate_status
        }

if __name__ == "__main__":
    app = EmotionDetector()

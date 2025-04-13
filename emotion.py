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
            "Sad": (0, 0, 255),    # Red
            "Stressed": (0, 0, 255)  # Red (kept for backward compatibility)
        }
        
        # Initialize emotion history
        self.emotion_history = deque(maxlen=history_size)
        self.valence_history = deque(maxlen=history_size)
        self.arousal_history = deque(maxlen=history_size)
        
    
    def calculate_emotion(self, expressions):
        """Calculate emotion based on facial expressions and blink rate.
        
        Args:
            expressions: Dictionary containing facial expression information
            blink_detected: Boolean indicating if a blink was detected
            
        Returns:
            tuple: (valence, arousal, emotion_name) values for the emotion
        """
        
        # Get key expressions for emotion detection
        is_frowning = expressions.get("eyebrows_frowning", False) or expressions.get("frowning", False)
        is_smiling = expressions.get("smiling", False)
        
        # Determine emotion based on the specified rules
        if is_frowning and not is_smiling:
            emotion_name = "Sad"
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
    
    def process_emotion(self, landmarks, expressions):
        """Process emotion based on facial landmarks and expressions.
        
        Args:
            landmarks: numpy array of facial landmarks
            expressions: Dictionary containing facial expression information
            
        Returns:
            dict: A dictionary containing:
                - valence: The calculated valence value (-1 to 1)
                - arousal: The calculated arousal value (-1 to 1)
                - emotion: The detected emotion name
                
        """
        # Calculate emotion
        valence, arousal, emotion_name = self.calculate_emotion(expressions)
        
        return {
            "valence": valence,
            "arousal": arousal,
            "emotion": emotion_name
        }

if __name__ == "__main__":
    app = EmotionDetector()

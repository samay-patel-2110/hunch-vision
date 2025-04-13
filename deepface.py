import cv2
from deepface import DeepFace
import time

def analyze_emotions_from_webcam():
    """
    Real-time facial expression analysis from webcam using DeepFace
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting facial expression analysis...")
    print("Press 'q' to quit")
    
    # For performance, we'll analyze every few frames
    frame_skip = 3
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        result_frame = frame.copy()
        
        # Only analyze every few frames to improve performance
        if frame_count % frame_skip == 0:
            try:
                # Analyze the frame using DeepFace
                analysis = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                # Handle both single and multiple face detections
                if not isinstance(analysis, list):
                    analysis = [analysis]
                
                # Process each detected face
                for face_data in analysis:
                    # Get face region
                    region = face_data.get('region', {})
                    if region:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        
                        # Draw rectangle around face
                        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Get emotion data
                        emotion = face_data['dominant_emotion']
                        emotion_score = face_data['emotion'][emotion]
                        
                        # Display emotion on frame
                        emotion_text = f"{emotion}: {emotion_score:.1f}%"
                        cv2.putText(
                            result_frame, 
                            emotion_text, 
                            (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, 
                            (0, 255, 0), 
                            2
                        )
                        
                        # Optional: Display all emotions with scores
                        y_offset = y + h + 20
                        for emo, score in face_data['emotion'].items():
                            if score > 5:  # Only show emotions with significant scores
                                text = f"{emo}: {score:.1f}%"
                                cv2.putText(
                                    result_frame,
                                    text,
                                    (x, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    1
                                )
                                y_offset += 20
            
            except Exception as e:
                print(f"Analysis error: {e}")
        
        # Add FPS information (frames analyzed per second)
        fps_text = f"Analyzing every {frame_skip} frames"
        cv2.putText(
            result_frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Display the result
        cv2.imshow('DeepFace Emotion Recognition', result_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam emotion analysis stopped")

if __name__ == "__main__":
    analyze_emotions_from_webcam()

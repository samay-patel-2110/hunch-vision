# A companion for long hours of laptop use!
Non contact respiratory and posture analysis and monitoring

A real-time monitoring system that detects emotions, eye states, blink patterns, and posture. The system provides feedback on emotional states and posture quality to help improve user well-being during computer usage.

## 1. System Setup

### Requirements

- **Python Version**: 3.8+ recommended
- **Required Libraries**:
  - OpenCV (`cv2`): 4.5.0+
  - MediaPipe: 0.8.9+
  - NumPy: 1.19.0+
  - psutil: 5.8.0+
  - pygame (optional, for audio feedback): 2.0.0+

### Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install opencv-python mediapipe numpy psutil pygame
```

### Running the Demo

The main script is `emotion_blink_v1_mediapipe.py`. Run it with the following command:

```bash
python final/main.py
```
Recommendation :
1) Look straight at the camera and keep netural face during calibration
2) Stay near to the laptop
3) Make sure upper body is comfortably covered in the camera view
4) Position camera at eye level, facing directly forward
5) Maintain 0.5-1.0 meter distance from camera

#### Command Line Arguments

- `--source`: Video source (default: `0` for webcam, or path to video file)
- `--width`: Window width (default: `1280`)
- `--height`: Window height (default: `720`)

Example with custom source and resolution:
```bash
python final/main.py --source 1 --width 1280 --height 720
```

### Controls

- Press `q` to quit the application
- Press `r` to reset blink counter
- Press `c` to recalibrate the system

## 2. Aim

1. **Eye State Detection**:
   - Blink detection through temporal analysis
   - Blink Counts

2. **Three way Emotion Detection**:
   - Rule-based algorithm using facial expressions
   - Valence-arousal model for emotion classification
   - Integrates eyebrow position, mouth shape, and eye state
   - Emotions : Relaxed, Angry, Happy 

3. **Posture Analysis**:
   - Shoulder position tracking using MediaPipe Pose
   - Point between chest tracking - Optical flow for smooth tracking between frames
   - Relative shoulder drop measurement for hunch detection

### Performance Benchmarks

Tests conducted on a typical system (MacBook M2 Pro, 16GB RAM, macOS 24.3.0):

- **CPU Usage**: 15-25% on average
- **Memory Usage**: ~300MB
- **Average FPS**: 20-25 FPS at 720p resolution
- **Startup Time**: ~2-3 seconds including initial calibration

### Limitations

#### Environmental Factors

- **Lighting**: Requires moderate to good lighting for accurate detection
  - Low light conditions reduce detection accuracy
  - Harsh backlighting can obscure facial features

- **Camera Position**: Best results when camera is at eye level
  - Extreme angles reduce landmark detection accuracy
  - Recommended distance: 0.5-1.0 meters from camera

- **Occlusions**: Performance degrades with facial occlusions
  - Glasses generally work but may affect eye state detection
  - Masks significantly reduce facial expression detection

#### Technical Limitations

1. **Low Frame Rate**:
   - Close other CPU-intensive applications
   - Reduce resolution using `--width` and `--height` parameters
   - Ensure adequate system cooling

2. **Poor Detection Accuracy**:
   - Improve lighting conditions
   - for shoulder detection, If the person moves away from the latop as optical flow is utilized
   - Adjust camera position to eye level
   - Recalibrate using the 'c' key
   - Remove facial occlusions if possible

### Validation Approach

The system was validated through multiple approaches:

1. **Manual Testing**:
   - Cross-checking detected states with known expressions
   - Comparing blink detection with manual counts
   - Testing posture detection with deliberate posture changes

2. **Performance Monitoring**:
   - Built-in metrics tracking CPU, memory, and frame rate
   - Visualization of detection confidence
   - Manual Verification by plotting and stability analysis

3. **Mitigation Strategies**:
   - Implemented temporal smoothing to reduce false positives
   - Added confidence levels for detected states
   - Calibration process for personalized baselines
   - Signal handlers for clean program termination

## 7. Future Improvements
- C++ for faster inference and image transport
- analysing and creating reports for wellbeing analytics
- Model fine-tuning on a custom dataset (approx 100 videos each of 5 minutes)

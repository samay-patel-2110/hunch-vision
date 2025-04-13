# Emotion and Posture Detection System

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
python final/emotion_blink_v1_mediapipe.py
```

#### Command Line Arguments

- `--source`: Video source (default: `0` for webcam, or path to video file)
- `--width`: Window width (default: `1280`)
- `--height`: Window height (default: `720`)

Example with custom source and resolution:
```bash
python final/emotion_blink_v1_mediapipe.py --source 1 --width 1280 --height 720
```

### Controls

- Press `q` to quit the application
- Press `r` to reset blink counter
- Press `c` to recalibrate the system

## 2. Model Choice & Pipeline

### Detection Pipeline

The system uses a multi-module approach:

1. **Face and Body Detection**: MediaPipe models
   - Face Mesh: 468 facial landmarks
   - Pose: Body keypoints including shoulders

2. **Eye State Detection**:
   - Custom algorithm based on eyelid distances
   - Blink detection through temporal analysis
   - Graph-based approach for reliable detection

3. **Emotion Detection**:
   - Rule-based algorithm using facial expressions
   - Valence-arousal model for emotion classification
   - Integrates eyebrow position, mouth shape, and eye state

4. **Posture Analysis**:
   - Shoulder position tracking using MediaPipe Pose
   - Optical flow for smooth tracking between frames
   - Relative shoulder drop measurement for hunch detection

### Library Choices

- **MediaPipe**: Used for robust facial landmark and pose detection
  - Provides consistent tracking even with partial occlusion
  - Optimized for real-time performance

- **OpenCV**: Used for video processing and visualization
  - Efficient frame manipulation
  - Drawing functions for analytics visualization

- **Custom Algorithms**: For state detection and classification
  - Calibration-based personalization
  - Weighted multi-point analysis
  - Temporal smoothing for stability

## 3. Performance Metrics

### System Requirements

The application is designed to run on standard consumer hardware:

- **Minimum Requirements**:
  - CPU: Dual-core 2.0 GHz
  - RAM: 4GB
  - Camera: Standard webcam (720p)
  - Storage: 100MB

- **Recommended Requirements**:
  - CPU: Quad-core 2.5 GHz+
  - RAM: 8GB+
  - Camera: HD webcam (1080p)
  - Storage: 500MB

### Performance Benchmarks

Tests conducted on a typical system (MacBook Pro, Intel Core i7, 16GB RAM, macOS 24.3.0):

- **CPU Usage**: 15-25% on average
- **Memory Usage**: ~300MB
- **Average FPS**: 25-30 FPS at 720p resolution
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

- **Processing Power**: Performance scales with CPU capability
  - Lower-end systems may experience reduced frame rates
  - Consider reducing resolution for better performance

- **Calibration Requirements**: System requires initial calibration
  - User must maintain neutral expression during calibration
  - Recalibration needed when lighting or position changes significantly

### Validation Approach

The system was validated through multiple approaches:

1. **Manual Testing**:
   - Cross-checking detected states with known expressions
   - Comparing blink detection with manual counts
   - Testing posture detection with deliberate posture changes

2. **Performance Monitoring**:
   - Built-in metrics tracking CPU, memory, and frame rate
   - Visualization of detection confidence
   - Historical data plotting for stability analysis

3. **Mitigation Strategies**:
   - Implemented temporal smoothing to reduce false positives
   - Added confidence levels for detected states
   - Calibration process for personalized baselines
   - Signal handlers for clean program termination

## 4. Usage Tips

### Optimal Setup

- Position camera at eye level, facing directly forward
- Ensure uniform, diffuse lighting on your face
- Avoid backgrounds with moving elements
- Maintain 0.5-1.0 meter distance from camera

### Calibration

The system requires an initial calibration period (5 seconds):
- Maintain a neutral expression
- Look directly at the camera
- Keep shoulders level and posture upright
- Remain still during the calibration process

### Interpreting Results

- **Eye State**: Displayed in top-right corner
- **Blink Count**: Shows number of detected blinks
- **Emotion**: Categorized as Relaxed, Happy, Angry, or Stressed
- **Posture**: Indicates Good Posture, Slight Hunch, Medium Hunch, or Severe Hunch

## 5. Troubleshooting

### Common Issues

1. **Low Frame Rate**:
   - Close other CPU-intensive applications
   - Reduce resolution using `--width` and `--height` parameters
   - Ensure adequate system cooling

2. **Poor Detection Accuracy**:
   - Improve lighting conditions
   - Adjust camera position to eye level
   - Recalibrate using the 'c' key
   - Remove facial occlusions if possible

3. **Program Crashes**:
   - Ensure all required libraries are installed
   - Update to the latest version of MediaPipe
   - Check camera access permissions

## 6. License and Attribution

This project uses the following open-source components:
- MediaPipe (Apache 2.0 License)
- OpenCV (BSD License)

## 7. Future Improvements

- Multi-face tracking and analysis
- Enhanced emotion detection using machine learning
- Posture correction suggestions
- Session analytics and reporting
- Integration with break reminder systems 
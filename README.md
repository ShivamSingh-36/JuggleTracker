# JuggleTracker
> **An Automated football juggle counter using computer vision and adaptive signal processing**
---
##  Features

### Core Capabilities
- **Real-time kick-up detection** with confidence scoring
- **Multi-modal tracking**: Combines YOLO object detection + pose estimation
- **Adaptive thresholding**: Self-adjusting parameters based on signal characteristics
- **Predictive interpolation**: Maintains tracking through occlusions (up to 15 frames)
- **Ground truth validation**: Precision/Recall/F1 metrics with frame tolerance

### Analysis & Visualization
- **Comparative method analysis**: Tests Conservative, Balanced, and Aggressive detection modes
- **Interactive exploration**: Hover annotations and manual verification tools
- **Export capabilities**: JSON results, CSV detections, annotated videos
---
##  Results

### Performance Metrics

| Method | Detections | Precision | Recall | F1 Score |
|--------|-----------|-----------|--------|----------|
| Conservative | 14 | 0.643 | 0.600 | 0.621 |
| **Balanced** | 16 | **0.750** | **0.800** | **0.774** |
| Aggressive | 20 | 0.650 | 0.867 | 0.743 |

**Key Findings:**
- **86.7% ground truth alignment** (±5 frame tolerance)
- **2.27 frame average offset** from expert annotations
- **100% ball detection rate** (no missed frames)
- **81.8% average confidence** across all detections

### Signal Quality
- **Ball Detection Rate**: 100%
- **Height Consistency**: 71.7%
- **Timing Consistency**: 64.8%
- **Overall Quality Score**: 84.5%
---
## 🎬 Demo

### Detection in Action
![Detection Timeline](assets/detection_timeline.png)
*Frame-by-frame confidence scores with detected kick-ups*

### Method Comparison
![Performance Comparison](assets/method_comparison.png)
*Comparative analysis of three detection strategies*

### Ground Truth Validation
![Ground Truth Alignment](assets/ground_truth_alignment.png)
*Gold lines indicate matches within ±5 frame tolerance*

---
## 🔬 Methodology

### System Architecture

```
Input Video
    ↓
┌─────────────────────────┐
│  YOLO Ball Detection    │ → Ball position (x, y)
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Pose Estimation        │ → Hip landmarks
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Height Calculation     │ → Ball-hip distance
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Signal Processing      │ → Smoothing + Normalization
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Peak Detection         │ → Kick-up events
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Adaptive Learning      │ → Threshold adjustment
└─────────────────────────┘
    ↓
  Output Results
```
### Algorithm Overview

1. **Ball Tracking**
   - YOLO v8 detects a football
   - Kalman-like interpolation for occlusions
   - Maintains tracking for up to 15 missing frames

2. **Pose Estimation**
   - MediaPipe-based hip landmark detection
   - Median filtering (10-frame window) for stability
   - Fallback to ground plane if pose unavailable

3. **Height Signal Generation**
   - Calculate vertical distance: `height = hip_y - ball_y`
   - Percentile-based normalization (5th-95th)
   - Moving average smoothing (4-8 frame window)

4. **Peak Detection**
   - Scipy `find_peaks` with dynamic thresholds
   - Minimum distance constraint (6-10 frames)
   - Prominence and height filtering

5. **Confidence Scoring**
   ```python
   confidence = (height_score + prominence_score + detection_score) / 3
   
   where:
     height_score = min(peak_height / 100, 1.0)
     prominence_score = min(peak_prominence / 20, 1.0)
     detection_score = 1.0 if ball_detected else 0.5
   ```

6. **Adaptive Thresholds** (Optional)
   - Monitors signal variability (std dev)
   - Adjusts prominence: `threshold += learning_rate × (target - current)`
   - Clamps values to prevent over-adaptation

### Detection Methods

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Smooth Window | 8 frames | 6 frames | 4 frames |
| Peak Distance | 10 frames | 8 frames | 6 frames |
| Prominence % | 40th | 30th | 20th |
| Height % | 50th | 40th | 30th |
| Confidence | 0.80 | 0.75 | 0.65 |

**Use Cases:**
- **Conservative**: High precision scenarios (competitions, official counts)
- **Balanced**: General purpose (training, skill assessment)
- **Aggressive**: High recall needed (research, exploratory analysis)

---

## 📈 Performance Comparison

### Precision-Recall Trade-off

```
Precision ↑                    Recall ↑
Conservative: 64.3%            Aggressive: 86.7%
Balanced: 75.0%                Balanced: 80.0%
Aggressive: 65.0%              Conservative: 60.0%

Best Overall: Balanced (F1 = 0.774)
```

### Timing Analysis

- **Average kick interval**: 0.55 seconds (12.6 frames @ 23 fps)
- **Consistency score**: 64.8% (std dev/mean)
- **Processing speed**: ~20-25 fps on CPU, ~40-50 fps on GPU
---
## 📁 Project Structure

```
kickup-detection/
│
├── juggle_detection.py          # Main detection script
├── analysis.ipynb                # Analysis notebook
├── README.md                     # This file
│
├── ground_truth.json             # Manual annotations (create this)
├── yolov8n.pt                    # YOLO weights (auto-downloaded)
│
├── juggle_results/               # Output directory (auto-created)
│   ├── *_results_*.json          # Full metrics & parameters
│   ├── *_detections_*.csv        # Detection timestamps
│   ├── *_comparison_*.csv        # Method comparison
│   └── *_annotated_*.mp4         # Annotated video
│
├── assets/                       # Documentation images
│   ├── detection_timeline.png
│   ├── method_comparison.png
│   └── ground_truth_alignment.png
│
└── videos/                       # Test videos (not included)
    ├── vidb.mp4
    └── (add your videos here)
```
## ⚙️ Configuration

### Detection Parameters

Edit `DETECTION_METHODS` dictionary in `kickup_detection.py`:

```python
DETECTION_METHODS = {
    'custom': {
        'SMOOTH_WINDOW': 5,              # Smoothing kernel size
        'PEAK_DISTANCE': 8,              # Min frames between kicks
        'PEAK_PROMINENCE_PERCENTILE': 25, # Signal prominence threshold
        'MIN_HEIGHT_PERCENTILE': 35,     # Minimum peak height
        'MIN_PROMINENCE_FLOOR': 4,       # Absolute minimum prominence
        'CONFIDENCE_THRESHOLD': 0.70     # Detection confidence cutoff
    }
}

ACTIVE_METHOD = 'custom'
```

### Adaptive Learning

```python
ENABLE_ADAPTIVE_THRESHOLDS = True
LEARNING_RATE = 0.1              # Adaptation speed (0.05-0.2)
ADAPTATION_WINDOW = 50           # Frames to consider for adaptation
```

### Video Output

```python
SAVE_ANNOTATED_VIDEO = True
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1200, 700  # Display resolution
```

---
## 🐛 Known Issues & Limitations

### Current Limitations
1. **Single-player only**: Does not handle multiple players simultaneously
2. **Fixed camera**: Assumes static camera position
3. **Lighting sensitivity**: Performance degrades in low-light conditions
4. **Ball type**: Optimized for standard football balls (white/traditional patterns)

### Troubleshooting

**Issue**: Low detection rate
```python
# Try switching to aggressive mode
ACTIVE_METHOD = 'aggressive'

# Or lower confidence threshold
CONFIDENCE_THRESHOLD = 0.65
```

**Issue**: Too many false positives
```python
# Use conservative mode
ACTIVE_METHOD = 'conservative'

# Or increase peak distance
PEAK_DISTANCE = 12
```

**Issue**: Ball not detected
- Ensure good lighting conditions
- Check that the ball is visible (not occluded)
- Verify the YOLO model is loaded correctly
- Try adjusting the YOLO confidence threshold
---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

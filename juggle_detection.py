import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import cvzone
from cvzone.PlotModule import LivePlot
from scipy.signal import find_peaks
from collections import deque
import json
from datetime import datetime
import os

VIDEO_PATH = 'vidb.mp4'
YOLO_WEIGHTS = 'yolov8n.pt'
BALL_CLASS_ID = 32
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1200, 700

GROUND_TRUTH_FILE = 'ground_truth.json'  # Format: {"video_name.mp4": [frame1, frame2, ...]}
ENABLE_GROUND_TRUTH_EVAL = True

DETECTION_METHODS = {
    'conservative': {
        'SMOOTH_WINDOW': 8,
        'PEAK_DISTANCE': 10,
        'PEAK_PROMINENCE_PERCENTILE': 40,
        'MIN_HEIGHT_PERCENTILE': 50,
        'MIN_PROMINENCE_FLOOR': 8,
        'CONFIDENCE_THRESHOLD': 0.80
    },
    'balanced': {  
        'SMOOTH_WINDOW': 6,
        'PEAK_DISTANCE': 8,
        'PEAK_PROMINENCE_PERCENTILE': 30,
        'MIN_HEIGHT_PERCENTILE': 40,
        'MIN_PROMINENCE_FLOOR': 5,
        'CONFIDENCE_THRESHOLD': 0.75
    },
    'aggressive': {
        'SMOOTH_WINDOW': 4,
        'PEAK_DISTANCE': 6,
        'PEAK_PROMINENCE_PERCENTILE': 20,
        'MIN_HEIGHT_PERCENTILE': 30,
        'MIN_PROMINENCE_FLOOR': 3,
        'CONFIDENCE_THRESHOLD': 0.65
    }
}

ACTIVE_METHOD = 'aggressive'
RUN_COMPARATIVE_ANALYSIS = True 
ENABLE_ADAPTIVE_THRESHOLDS = True
LEARNING_RATE = 0.1 
ADAPTATION_WINDOW = 50  
params = DETECTION_METHODS[ACTIVE_METHOD]
SMOOTH_WINDOW = params['SMOOTH_WINDOW']
PEAK_DISTANCE = params['PEAK_DISTANCE']
PEAK_PROMINENCE_PERCENTILE = params['PEAK_PROMINENCE_PERCENTILE']
MIN_HEIGHT_PERCENTILE = params['MIN_HEIGHT_PERCENTILE']
MIN_PROMINENCE_FLOOR = params['MIN_PROMINENCE_FLOOR']
CONFIDENCE_THRESHOLD = params['CONFIDENCE_THRESHOLD']

HISTORY_SIZE = 30
SAVE_ANNOTATED_VIDEO = True
EXPORT_RESULTS = True
OUTPUT_DIR = 'kickup_results'

adaptive_state = {
    'prominence_threshold': MIN_PROMINENCE_FLOOR,
    'height_threshold': MIN_HEIGHT_PERCENTILE,
    'recent_false_positives': deque(maxlen=ADAPTATION_WINDOW),
    'recent_true_positives': deque(maxlen=ADAPTATION_WINDOW)
}

def load_ground_truth(video_path, gt_file):
    """Load ground truth kick-up frames if available"""
    if not os.path.exists(gt_file):
        print(f"⚠ Ground truth file not found: {gt_file}")
        print("  To enable evaluation, create a JSON file with format:")
        print('  {"video_name.mp4": [frame1, frame2, frame3, ...]}')
        return None
    
    try:
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
        
        video_name = os.path.basename(video_path)
        if video_name in gt_data:
            frames = sorted(gt_data[video_name])
            print(f"✓ Loaded ground truth: {len(frames)} kick-ups")
            return frames
        else:
            print(f"⚠ No ground truth for {video_name} in {gt_file}")
            return None
    except Exception as e:
        print(f"⚠ Error loading ground truth: {e}")
        return None

def calculate_metrics(detected_frames, ground_truth_frames, tolerance=5):
    """Calculate precision, recall, F1 score with frame tolerance"""
    if not ground_truth_frames:
        return None
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    matched_gt = set()
    matched_det = set()
    
    # Match detections to ground truth
    for det_frame in detected_frames:
        matched = False
        for gt_frame in ground_truth_frames:
            if abs(det_frame - gt_frame) <= tolerance and gt_frame not in matched_gt:
                tp += 1
                matched_gt.add(gt_frame)
                matched_det.add(det_frame)
                matched = True
                break
        if not matched:
            fp += 1
    
    # Count missed ground truths
    fn = len(ground_truth_frames) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tolerance_frames': tolerance,
        'matched_detections': sorted(matched_det),
        'missed_ground_truth': sorted(set(ground_truth_frames) - matched_gt),
        'false_detections': sorted(set(detected_frames) - matched_det)
    }

def adapt_thresholds(adaptive_state, current_signal, recent_detections):
    """Dynamically adjust thresholds based on signal characteristics"""
    if not ENABLE_ADAPTIVE_THRESHOLDS or len(current_signal) < ADAPTATION_WINDOW:
        return adaptive_state['prominence_threshold'], adaptive_state['height_threshold']
    
    # Analyze signal variability
    signal_std = np.std(current_signal[-ADAPTATION_WINDOW:])
    signal_mean = np.mean(current_signal[-ADAPTATION_WINDOW:])
    
    # Calculate detection density (detections per frame)
    detection_density = len(recent_detections) / ADAPTATION_WINDOW if recent_detections else 0
    
    # Adapt prominence threshold based on signal noise
    target_prominence = MIN_PROMINENCE_FLOOR + (signal_std / 10)
    adaptive_state['prominence_threshold'] += LEARNING_RATE * (target_prominence - adaptive_state['prominence_threshold'])
    
    # Adapt height threshold based on detection density
    if detection_density > 0.15:  # Too many detections - increase threshold
        adaptive_state['height_threshold'] += LEARNING_RATE * 5
    elif detection_density < 0.05:  # Too few detections - decrease threshold
        adaptive_state['height_threshold'] -= LEARNING_RATE * 5
    
    # Clamp values
    adaptive_state['prominence_threshold'] = np.clip(adaptive_state['prominence_threshold'], 2, 15)
    adaptive_state['height_threshold'] = np.clip(adaptive_state['height_threshold'], 20, 70)
    
    return adaptive_state['prominence_threshold'], adaptive_state['height_threshold']

def run_detection_with_params(video_path, params_dict, ground_truth=None):
    """Run detection with specific parameter set (for comparative analysis)"""
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    model = YOLO(YOLO_WEIGHTS)
    
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    height_history = deque(maxlen=HISTORY_SIZE)
    ball_y_history = deque(maxlen=HISTORY_SIZE)
    hip_y_history = deque(maxlen=HISTORY_SIZE)
    full_height_history = []
    
    kickup_detections = []
    frame_count = 0
    kickup_count = 0
    last_known_ball_y = None
    frames_since_detection = 0
    last_peak_frame = -999
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Ball detection (same as original)
        results = model(img, stream=False, verbose=False)
        ball_detected = False
        current_ball_y = None
        ball_confidence = 0
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == BALL_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_ball_y = y1 + (y2 - y1) // 2
                    ball_confidence = float(box.conf[0])
                    ball_detected = True
                    frames_since_detection = 0
                    last_known_ball_y = current_ball_y
                    break
            if ball_detected:
                break
        
        if not ball_detected and last_known_ball_y is not None:
            frames_since_detection += 1
            if frames_since_detection <= 15:
                if len(ball_y_history) >= 3:
                    recent_positions = list(ball_y_history)[-3:]
                    velocity = (recent_positions[-1] - recent_positions[0]) / 2
                    current_ball_y = last_known_ball_y + velocity * frames_since_detection
                    current_ball_y = np.clip(current_ball_y, 0, frame_height)
                else:
                    current_ball_y = last_known_ball_y
        
        # Pose detection (same as original)
        img = detector.findPose(img, draw=False)
        lmList, _ = detector.findPosition(img, draw=False)
        
        current_hip_y = None
        if lmList:
            left_hip = lmList[23][:2]
            right_hip = lmList[24][:2]
            current_hip_y = (left_hip[1] + right_hip[1]) / 2
            hip_y_history.append(current_hip_y)
        
        height_value = None
        if current_ball_y is not None:
            ball_y_history.append(current_ball_y)
            if current_hip_y is not None and len(hip_y_history) >= 3:
                median_hip = np.median(list(hip_y_history)[-10:])
                height_value = median_hip - current_ball_y
            else:
                height_value = frame_height - current_ball_y
            
            height_history.append(height_value)
            full_height_history.append(height_value)
        
        # Peak detection with current params
        if len(height_history) >= 15:
            height_array = np.array(height_history)
            p5 = np.percentile(height_array, 5)
            p95 = np.percentile(height_array, 95)
            height_range = p95 - p5 + 1e-6
            height_normalized = 100 * (height_array - p5) / height_range
            height_normalized = np.clip(height_normalized, 0, 100)
            
            if len(height_normalized) >= params_dict['SMOOTH_WINDOW']:
                kernel = np.ones(params_dict['SMOOTH_WINDOW']) / params_dict['SMOOTH_WINDOW']
                height_smooth = np.convolve(height_normalized, kernel, mode='same')
            else:
                height_smooth = height_normalized
            
            if len(height_smooth) >= 20:
                signal_diff = np.abs(np.diff(height_smooth))
                prominence_threshold = np.percentile(signal_diff, params_dict['PEAK_PROMINENCE_PERCENTILE'])
                min_peak_height = np.percentile(height_smooth, params_dict['MIN_HEIGHT_PERCENTILE'])
                
                peaks, props = find_peaks(
                    height_smooth,
                    distance=params_dict['PEAK_DISTANCE'],
                    prominence=max(prominence_threshold, params_dict['MIN_PROMINENCE_FLOOR']),
                    height=min_peak_height
                )
                
                if len(peaks) > 0:
                    latest_peak_idx = peaks[-1]
                    latest_peak_frame = frame_count - (len(height_smooth) - latest_peak_idx)
                    
                    if latest_peak_frame > last_peak_frame + params_dict['PEAK_DISTANCE']:
                        peak_height = height_smooth[latest_peak_idx]
                        peak_prominence = props['prominences'][-1] if 'prominences' in props else 0
                        
                        height_conf = min(peak_height / 100, 1.0)
                        prominence_conf = min(peak_prominence / 20, 1.0)
                        ball_detect_conf = 1.0 if ball_detected else 0.5
                        overall_confidence = (height_conf + prominence_conf + ball_detect_conf) / 3
                        
                        if overall_confidence >= params_dict['CONFIDENCE_THRESHOLD']:
                            kickup_count += 1
                            last_peak_frame = latest_peak_frame
                            kickup_detections.append({
                                'frame': frame_count,
                                'time_sec': frame_count / fps,
                                'confidence': overall_confidence,
                                'height': peak_height,
                                'ball_detected': ball_detected
                            })
    
    cap.release()
    
    detected_frames = [d['frame'] for d in kickup_detections]
    metrics = calculate_metrics(detected_frames, ground_truth) if ground_truth else None
    
    return {
        'count': kickup_count,
        'detections': kickup_detections,
        'detected_frames': detected_frames,
        'metrics': metrics
    }

# Create output directory
if SAVE_ANNOTATED_VIDEO or EXPORT_RESULTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# Load ground truth
ground_truth_frames = None
if ENABLE_GROUND_TRUTH_EVAL:
    ground_truth_frames = load_ground_truth(VIDEO_PATH, GROUND_TRUTH_FILE)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video {VIDEO_PATH}")

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if SAVE_ANNOTATED_VIDEO:
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_name}_annotated_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

detector = PoseDetector()
model = YOLO(YOLO_WEIGHTS)
plotY = LivePlot(w=1200, h=400, yLimit=[0, 100], invert=True)

height_history = deque(maxlen=HISTORY_SIZE)
ball_y_history = deque(maxlen=HISTORY_SIZE)
hip_y_history = deque(maxlen=HISTORY_SIZE)
full_height_history = []
full_ball_y_history = []
full_hip_y_history = []

kickup_detections = []
frame_count = 0
kickup_count = 0
last_known_ball_y = None
frames_since_detection = 0
MAX_INTERPOLATION_FRAMES = 15
recent_peaks = deque(maxlen=5)
last_peak_frame = -999

cv2.namedWindow("Kick-up Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Kick-up Counter", DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("="*60)
print("FOOTBALL KICK-UP DETECTION SYSTEM v3.0 [ENHANCED]")
print("="*60)
print(f"Video: {VIDEO_PATH}")
print(f"Detection Method: {ACTIVE_METHOD.upper()}")
print(f"Adaptive Thresholds: {'ENABLED' if ENABLE_ADAPTIVE_THRESHOLDS else 'DISABLED'}")
print(f"Ground Truth Eval: {'ENABLED' if ground_truth_frames else 'DISABLED'}")
print(f"Comparative Analysis: {'ENABLED' if RUN_COMPARATIVE_ANALYSIS else 'DISABLED'}")
print("="*60)
print("\nStarting detection... Press 'q' to quit\n")

while True:
    success, img = cap.read()
    if not success:
        print(f"\nVideo finished. Total Kick-ups: {kickup_count}")
        break

    frame_count += 1
    
    results = model(img, stream=False, verbose=False)
    ball_detected = False
    current_ball_y = None
    ball_confidence = 0
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                current_ball_y = cy
                ball_confidence = float(box.conf[0])
                ball_detected = True
                frames_since_detection = 0
                last_known_ball_y = cy
                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), colorR=(0,255,255), t=2)
                cvzone.putTextRect(img, f"BALL {ball_confidence:.2f}", (x1, y1-10), 
                                 colorR=(0,255,255), colorT=(0,0,0), 
                                 scale=1.2, thickness=2, offset=5)
                break
        if ball_detected:
            break
    
    if not ball_detected and last_known_ball_y is not None:
        frames_since_detection += 1
        if frames_since_detection <= MAX_INTERPOLATION_FRAMES:
            if len(ball_y_history) >= 3:
                recent_positions = list(ball_y_history)[-3:]
                velocity = (recent_positions[-1] - recent_positions[0]) / 2
                current_ball_y = last_known_ball_y + velocity * frames_since_detection
                current_ball_y = np.clip(current_ball_y, 0, frame_height)
            else:
                current_ball_y = last_known_ball_y
            cvzone.putTextRect(img, "BALL (estimated)", (50, 150), 
                             colorR=(255,100,100), colorT=(255,255,255), 
                             scale=1.2, thickness=2, offset=10)
    
    img = detector.findPose(img, draw=False)
    lmList, _ = detector.findPosition(img, draw=False)
    
    current_hip_y = None
    if lmList:
        left_hip = lmList[23][:2]
        right_hip = lmList[24][:2]
        current_hip_y = (left_hip[1] + right_hip[1]) / 2
        hip_y_history.append(current_hip_y)
        full_hip_y_history.append(current_hip_y)
        cv2.circle(img, (int(left_hip[0]), int(left_hip[1])), 8, (255,0,255), cv2.FILLED)
        cv2.circle(img, (int(right_hip[0]), int(right_hip[1])), 8, (255,0,255), cv2.FILLED)
    
    height_value = None
    if current_ball_y is not None:
        ball_y_history.append(current_ball_y)
        full_ball_y_history.append(current_ball_y)
        
        if current_hip_y is not None and len(hip_y_history) >= 3:
            median_hip = np.median(list(hip_y_history)[-10:])
            height_value = median_hip - current_ball_y
        else:
            height_value = frame_height - current_ball_y
        
        height_history.append(height_value)
        full_height_history.append(height_value)
    
    if len(height_history) >= 15:
        height_array = np.array(height_history)
        p5 = np.percentile(height_array, 5)
        p95 = np.percentile(height_array, 95)
        height_range = p95 - p5 + 1e-6
        height_normalized = 100 * (height_array - p5) / height_range
        height_normalized = np.clip(height_normalized, 0, 100)
        
        if len(height_normalized) >= SMOOTH_WINDOW:
            kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            height_smooth = np.convolve(height_normalized, kernel, mode='same')
        else:
            height_smooth = height_normalized
        
        if len(height_smooth) >= 20:
            # Apply adaptive thresholds
            adaptive_prom, adaptive_height = adapt_thresholds(
                adaptive_state, height_smooth, list(recent_peaks)
            )
            
            signal_diff = np.abs(np.diff(height_smooth))
            prominence_threshold = np.percentile(signal_diff, PEAK_PROMINENCE_PERCENTILE)
            
            # Use adaptive or fixed thresholds
            final_prom = adaptive_prom if ENABLE_ADAPTIVE_THRESHOLDS else max(prominence_threshold, MIN_PROMINENCE_FLOOR)
            final_height = adaptive_height if ENABLE_ADAPTIVE_THRESHOLDS else np.percentile(height_smooth, MIN_HEIGHT_PERCENTILE)
            
            peaks, props = find_peaks(
                height_smooth,
                distance=PEAK_DISTANCE,
                prominence=final_prom,
                height=final_height
            )
            
            if len(peaks) > 0:
                latest_peak_idx = peaks[-1]
                latest_peak_frame = frame_count - (len(height_smooth) - latest_peak_idx)
                
                if latest_peak_frame > last_peak_frame + PEAK_DISTANCE:
                    peak_height = height_smooth[latest_peak_idx]
                    peak_prominence = props['prominences'][-1] if 'prominences' in props else 0
                    
                    height_conf = min(peak_height / 100, 1.0)
                    prominence_conf = min(peak_prominence / 20, 1.0)
                    ball_detect_conf = 1.0 if ball_detected else 0.5
                    overall_confidence = (height_conf + prominence_conf + ball_detect_conf) / 3
                    
                    if overall_confidence >= CONFIDENCE_THRESHOLD:
                        kickup_count += 1
                        last_peak_frame = latest_peak_frame
                        recent_peaks.append(latest_peak_frame)
                        
                        kickup_detections.append({
                            'frame': frame_count,
                            'time_sec': frame_count / fps,
                            'confidence': overall_confidence,
                            'height': peak_height,
                            'ball_detected': ball_detected
                        })
                        
                        print(f"Kick-up #{kickup_count} | Frame {frame_count} | Confidence: {overall_confidence:.2f}")
        
        current_height = height_smooth[-1] if len(height_smooth) > 0 else 50
        frames_since_last_peak = frame_count - last_peak_frame
        
        if frames_since_last_peak < 10:
            plot_color = (0, 255, 0)
        elif ball_detected:
            plot_color = (0, 200, 200)
        else:
            plot_color = (100, 100, 255)
        
        imgPlot = plotY.update(current_height, color=plot_color)
    else:
        imgPlot = plotY.update(50, color=(128, 128, 128))
    
    cvzone.putTextRect(img, f'KICK-UPS: {kickup_count}', (50, 80), 
                      scale=3, thickness=3, colorR=(0,255,0), 
                      colorT=(0,0,0), offset=20)
    
    status_text = "TRACKING" if ball_detected else f"ESTIMATING ({frames_since_detection}f)"
    status_color = (0,255,0) if ball_detected else (255,150,0)
    cvzone.putTextRect(img, status_text, (frame_width - 250, 80), 
                      scale=1.5, thickness=2, colorR=status_color, 
                      colorT=(0,0,0), offset=10)
    
    img_resized = cv2.resize(img, (imgPlot.shape[1], imgPlot.shape[0]))
    imgStack = cvzone.stackImages([img_resized, imgPlot], 2, 1)
    
    if SAVE_ANNOTATED_VIDEO:
        out.write(cv2.resize(imgStack, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
    
    cv2.imshow("Kick-up Counter", imgStack)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if SAVE_ANNOTATED_VIDEO:
    out.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}")
print(f"PRIMARY DETECTION COMPLETE: {kickup_count} kick-ups")
print(f"{'='*60}")

# Calculate metrics for primary method
detected_frames = [d['frame'] for d in kickup_detections]
primary_metrics = calculate_metrics(detected_frames, ground_truth_frames) if ground_truth_frames else None

if primary_metrics:
    print(f"\n{'='*60}")
    print(f"QUANTITATIVE EVALUATION - {ACTIVE_METHOD.upper()} METHOD")
    print(f"{'='*60}")
    print(f"Ground Truth Kick-ups: {len(ground_truth_frames)}")
    print(f"Detected Kick-ups: {kickup_count}")
    print(f"\nPerformance Metrics (±{primary_metrics['tolerance_frames']} frame tolerance):")
    print(f"  Precision: {primary_metrics['precision']:.3f} ({primary_metrics['true_positives']}/{primary_metrics['true_positives']+primary_metrics['false_positives']})")
    print(f"  Recall:    {primary_metrics['recall']:.3f} ({primary_metrics['true_positives']}/{primary_metrics['true_positives']+primary_metrics['false_negatives']})")
    print(f"  F1 Score:  {primary_metrics['f1_score']:.3f}")
    print(f"\nDetection Breakdown:")
    print(f"  True Positives:  {primary_metrics['true_positives']}")
    print(f"  False Positives: {primary_metrics['false_positives']}")
    print(f"  False Negatives: {primary_metrics['false_negatives']}")
    
    if primary_metrics['false_detections']:
        print(f"\n⚠ False Positive Frames: {primary_metrics['false_detections'][:10]}" + 
              ("..." if len(primary_metrics['false_detections']) > 10 else ""))
    if primary_metrics['missed_ground_truth']:
        print(f"⚠ Missed Ground Truth: {primary_metrics['missed_ground_truth'][:10]}" +
              ("..." if len(primary_metrics['missed_ground_truth']) > 10 else ""))

# Comparative analysis
comparative_results = {}
if RUN_COMPARATIVE_ANALYSIS and ground_truth_frames:
    print(f"\n{'='*60}")
    print("RUNNING COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    for method_name, method_params in DETECTION_METHODS.items():
        if method_name == ACTIVE_METHOD:
            comparative_results[method_name] = {
                'count': kickup_count,
                'detections': kickup_detections,
                'detected_frames': detected_frames,
                'metrics': primary_metrics
            }
            print(f"✓ {method_name}: Using primary results")
        else:
            print(f"  Running {method_name} method...")
            result = run_detection_with_params(VIDEO_PATH, method_params, ground_truth_frames)
            comparative_results[method_name] = result
            print(f"  ✓ {method_name}: {result['count']} detections, F1={result['metrics']['f1_score']:.3f}")
    
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Count':<8} {'Precision':<12} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 60)
    
    best_f1 = 0
    best_method = None
    
    for method_name, result in comparative_results.items():
        metrics = result['metrics']
        marker = "→" if method_name == ACTIVE_METHOD else " "
        print(f"{marker} {method_name:<14} {result['count']:<8} "
              f"{metrics['precision']:<12.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_method = method_name
    
    print(f"\n✓ Best Method: {best_method.upper()} (F1={best_f1:.3f})")
    
    if best_method != ACTIVE_METHOD:
        print(f"⚠ Consider switching to '{best_method}' method for better accuracy")

# Post-processing analysis
print("\n" + "="*60)
print("POST-PROCESSING ANALYSIS")
print("="*60)

ball_y_array = np.array(full_ball_y_history)
height_array = np.array(full_height_history)

if len(height_array) >= 30:
    p5 = np.percentile(height_array, 5)
    p95 = np.percentile(height_array, 95)
    height_range = p95 - p5 + 1e-6
    height_normalized = 100 * (height_array - p5) / height_range
    height_normalized = np.clip(height_normalized, 0, 100)
    
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    height_smooth = np.convolve(height_normalized, kernel, mode='same')
    
    signal_diff = np.abs(np.diff(height_smooth))
    prominence_threshold = np.percentile(signal_diff, PEAK_PROMINENCE_PERCENTILE)
    min_peak_height = np.percentile(height_smooth, MIN_HEIGHT_PERCENTILE)
    
    peaks, props = find_peaks(
        height_smooth,
        distance=PEAK_DISTANCE,
        prominence=max(prominence_threshold, MIN_PROMINENCE_FLOOR),
        height=min_peak_height
    )
    
    # Calculate statistics
    if len(peaks) > 0:
        peak_heights = height_smooth[peaks]
        avg_height = np.mean(peak_heights)
        std_height = np.std(peak_heights)
        max_height = np.max(peak_heights)
        min_height = np.min(peak_heights)
        
        if len(peaks) > 1:
            kick_intervals = np.diff(peaks)
            avg_interval = np.mean(kick_intervals)
            consistency_score = 100 * (1 - np.std(kick_intervals) / (np.mean(kick_intervals) + 1))
        else:
            avg_interval = 0
            consistency_score = 0
    else:
        avg_height = std_height = max_height = min_height = 0
        avg_interval = consistency_score = 0
    
    # Export results
    if EXPORT_RESULTS:
        # Prepare comparative results for export
        comparative_summary = {}
        if RUN_COMPARATIVE_ANALYSIS and comparative_results:
            for method_name, result in comparative_results.items():
                comparative_summary[method_name] = {
                    'count': result['count'],
                    'metrics': result['metrics'] if result['metrics'] else {}
                }
        
        results_data = {
            'video_file': VIDEO_PATH,
            'analysis_timestamp': datetime.now().isoformat(),
            'detection_method': ACTIVE_METHOD,
            'adaptive_thresholds_enabled': ENABLE_ADAPTIVE_THRESHOLDS,
            'detection_parameters': {
                'smooth_window': SMOOTH_WINDOW,
                'peak_distance': PEAK_DISTANCE,
                'peak_prominence_percentile': PEAK_PROMINENCE_PERCENTILE,
                'min_height_percentile': MIN_HEIGHT_PERCENTILE,
                'confidence_threshold': CONFIDENCE_THRESHOLD
            },
            'ground_truth': {
                'available': ground_truth_frames is not None,
                'count': len(ground_truth_frames) if ground_truth_frames else None,
                'frames': ground_truth_frames if ground_truth_frames else None
            },
            'summary': {
                'total_frames': frame_count,
                'video_duration_sec': frame_count / fps,
                'fps': fps,
                'realtime_count': kickup_count,
                'postanalysis_count': len(peaks),
                'ball_detection_rate': float(len(ball_y_array)/frame_count)
            },
            'performance_metrics': primary_metrics if primary_metrics else {},
            'comparative_analysis': comparative_summary,
            'best_method': best_method if RUN_COMPARATIVE_ANALYSIS and comparative_results else None,
            'statistics': {
                'avg_kick_height': float(avg_height),
                'std_kick_height': float(std_height),
                'max_kick_height': float(max_height),
                'min_kick_height': float(min_height),
                'avg_frames_between_kicks': float(avg_interval),
                'consistency_score': float(consistency_score)
            },
            'detections': kickup_detections,
            'adaptive_learning_final_state': {
                'prominence_threshold': float(adaptive_state['prominence_threshold']),
                'height_threshold': float(adaptive_state['height_threshold'])
            } if ENABLE_ADAPTIVE_THRESHOLDS else {}
        }
        
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # CSV export
        csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_detections_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("Kickup_Number,Frame,Time_Sec,Confidence,Height,Ball_Detected\n")
            for i, det in enumerate(kickup_detections, 1):
                f.write(f"{i},{det['frame']},{det['time_sec']:.2f},"
                       f"{det['confidence']:.3f},{det['height']:.2f},{det['ball_detected']}\n")
        
        # Comparative analysis CSV
        if RUN_COMPARATIVE_ANALYSIS and comparative_results:
            comp_csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison_{timestamp}.csv")
            with open(comp_csv_path, 'w') as f:
                f.write("Method,Count,Precision,Recall,F1_Score,True_Positives,False_Positives,False_Negatives\n")
                for method_name, result in comparative_results.items():
                    m = result['metrics']
                    f.write(f"{method_name},{result['count']},{m['precision']:.4f},"
                           f"{m['recall']:.4f},{m['f1_score']:.4f},"
                           f"{m['true_positives']},{m['false_positives']},{m['false_negatives']}\n")
            print(f"\n✓ Results exported:")
            print(f"  - JSON: {json_path}")
            print(f"  - CSV: {csv_path}")
            print(f"  - Comparison: {comp_csv_path}")
        else:
            print(f"\n✓ Results exported:")
            print(f"  - JSON: {json_path}")
            print(f"  - CSV: {csv_path}")
    
    if SAVE_ANNOTATED_VIDEO:
        print(f"  - Video: {output_video_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    print(f"\nDetection Counts:")
    print(f"  Real-time (with confidence): {kickup_count}")
    print(f"  Post-analysis (all peaks): {len(peaks)}")
    print(f"  Match: {'✓ YES' if kickup_count == len(peaks) else '✗ NO'}")
    
    if kickup_count != len(peaks):
        diff = abs(kickup_count - len(peaks))
        print(f"  Discrepancy: {diff} kick-up(s)")
    
    print(f"\nHeight Analysis:")
    print(f"  Average: {avg_height:.1f}")
    print(f"  Std Dev: {std_height:.1f}")
    print(f"  Range: {min_height:.1f} - {max_height:.1f}")
    
    print(f"\nTiming Analysis:")
    print(f"  Avg interval: {avg_interval:.1f} frames ({avg_interval/fps:.2f}s)")
    print(f"  Consistency: {consistency_score:.1f}%")
    
    print(f"\nSignal Quality:")
    print(f"  Ball detection rate: {(len(ball_y_array)/frame_count)*100:.1f}%")
    print(f"  Frames analyzed: {len(height_array)}")
    print(f"  Signal range: {height_range:.1f} px")
    
    if ENABLE_ADAPTIVE_THRESHOLDS:
        print(f"\nAdaptive Learning (Final State):")
        print(f"  Prominence threshold: {adaptive_state['prominence_threshold']:.2f}")
        print(f"  Height threshold: {adaptive_state['height_threshold']:.1f}")
    
    print("="*60)
    
    # Matplotlib visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    
    manual_count = [0]
    
    # ===== GRAPH 1: Raw vs Normalized =====
    fig1 = plt.figure(figsize=(16, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(height_array, alpha=0.4, label='Raw Height', color='gray', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(height_normalized, label='Normalized (0-100)', color='blue', linewidth=2.5)
    ax1.set_xlabel('Frame Index', fontsize=14)
    ax1.set_ylabel('Raw Height (px)', color='gray', fontsize=14)
    ax1_twin.set_ylabel('Normalized Height', color='blue', fontsize=14)
    ax1.set_title('Signal Normalization: Raw vs Normalized', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=12)
    ax1_twin.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    ax1_twin.tick_params(labelsize=12)
    fig1.tight_layout()
    
    # ===== GRAPH 2: Detection Results with Ground Truth =====
    fig2 = plt.figure(figsize=(16, 8))
    ax2 = fig2.add_subplot(111)
    ax2.plot(height_smooth, label='Smoothed Signal', color='orange', linewidth=3)
    
    # Plot detected peaks
    for i, peak in enumerate(peaks, 1):
        ax2.plot(peak, height_smooth[peak], 'x', markersize=20, 
                markeredgewidth=4, color='red')
        ax2.text(peak, height_smooth[peak] + 7, str(i), 
                fontsize=13, ha='center', color='darkred', fontweight='bold')
    
    # Plot ground truth if available
    if ground_truth_frames:
        for gt_frame in ground_truth_frames:
            if gt_frame < len(height_smooth):
                ax2.axvline(gt_frame, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax2.plot([], [], 'g--', linewidth=2, label='Ground Truth')
    
    ax2.axhline(min_peak_height, color='red', linestyle='--', alpha=0.6, linewidth=2.5,
               label=f'Min Height Threshold ({MIN_HEIGHT_PERCENTILE}th percentile)')
    
    for peak in peaks:
        ax2.axvspan(peak-PEAK_DISTANCE//2, peak+PEAK_DISTANCE//2, 
                   alpha=0.2, color='green')
    
    match_status = "MATCH ✓" if kickup_count == len(peaks) else f"MISMATCH ({abs(kickup_count-len(peaks))} diff)"
    if primary_metrics:
        title_suffix = f" | F1={primary_metrics['f1_score']:.3f}"
    else:
        title_suffix = ""
    
    ax2.set_xlabel('Frame Index (Click on peaks to inspect)', fontsize=14)
    ax2.set_ylabel('Normalized Height (0-100)', fontsize=14)
    ax2.set_title(f'Kick-up Detection Results: Real-time={kickup_count}, Post-analysis={len(peaks)} | {match_status}{title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    
    # Manual counter buttons
    def increment_manual(event):
        manual_count[0] += 1
        manual_text.set_text(f'Manual Count: {manual_count[0]}')
        fig2.canvas.draw()
    
    def decrement_manual(event):
        manual_count[0] = max(0, manual_count[0] - 1)
        manual_text.set_text(f'Manual Count: {manual_count[0]}')
        fig2.canvas.draw()
    
    def onclick(event):
        if event.inaxes == ax2:
            frame_x = int(event.xdata)
            for peak in peaks:
                if abs(peak - frame_x) <= PEAK_DISTANCE:
                    print(f"✓ Peak #{list(peaks).index(peak)+1} at frame {peak}")
                    return
            print(f"✗ No peak at frame {frame_x}")
    
    fig2.canvas.mpl_connect('button_press_event', onclick)
    
    ax_button_plus = fig2.add_axes([0.52, 0.02, 0.04, 0.03])
    ax_button_minus = fig2.add_axes([0.47, 0.02, 0.04, 0.03])
    btn_plus = Button(ax_button_plus, '+1', color='lightgreen', hovercolor='green')
    btn_minus = Button(ax_button_minus, '-1', color='lightcoral', hovercolor='red')
    btn_plus.on_clicked(increment_manual)
    btn_minus.on_clicked(decrement_manual)
    
    manual_text = ax2.text(0.5, 1.08, f'Manual Count: {manual_count[0]}', 
                          transform=ax2.transAxes, fontsize=16, 
                          ha='center', color='blue', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, pad=1))
    
    fig2.tight_layout()
    
    # ===== GRAPH 3: Velocity Analysis =====
    fig3 = plt.figure(figsize=(16, 6))
    ax3 = fig3.add_subplot(111)
    signal_gradient = np.gradient(height_smooth)
    ax3.plot(signal_gradient, color='purple', linewidth=2.5, label='Ball Velocity')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax3.fill_between(range(len(signal_gradient)), signal_gradient, 0, 
                    where=(signal_gradient > 0), alpha=0.4, color='green', label='Rising (ball going up)')
    ax3.fill_between(range(len(signal_gradient)), signal_gradient, 0, 
                    where=(signal_gradient < 0), alpha=0.4, color='red', label='Falling (ball going down)')
    ax3.set_xlabel('Frame Index', fontsize=14)
    ax3.set_ylabel('Height Change Rate (velocity)', fontsize=14)
    ax3.set_title('Ball Velocity Analysis', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(loc='best', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12)
    fig3.tight_layout()
    
    # ===== GRAPH 4: Comparative Analysis =====
    if RUN_COMPARATIVE_ANALYSIS and comparative_results:
        fig4 = plt.figure(figsize=(12, 6))
        ax4 = fig4.add_subplot(111)
        
        methods = list(comparative_results.keys())
        f1_scores = [comparative_results[m]['metrics']['f1_score'] for m in methods]
        precisions = [comparative_results[m]['metrics']['precision'] for m in methods]
        recalls = [comparative_results[m]['metrics']['recall'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax4.bar(x - width, precisions, width, label='Precision', color='skyblue')
        bars2 = ax4.bar(x, recalls, width, label='Recall', color='lightcoral')
        bars3 = ax4.bar(x + width, f1_scores, width, label='F1 Score', color='lightgreen')
        
        # Highlight active method
        active_idx = methods.index(ACTIVE_METHOD)
        bars1[active_idx].set_edgecolor('black')
        bars1[active_idx].set_linewidth(3)
        bars2[active_idx].set_edgecolor('black')
        bars2[active_idx].set_linewidth(3)
        bars3[active_idx].set_edgecolor('black')
        bars3[active_idx].set_linewidth(3)
        
        ax4.set_xlabel('Detection Method', fontsize=14)
        ax4.set_ylabel('Score', fontsize=14)
        ax4.set_title('Comparative Performance Analysis (Active method has black border)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.capitalize() for m in methods], fontsize=12)
        ax4.legend(fontsize=12)
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(labelsize=12)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        fig4.tight_layout()
    
    plt.show()
    
    print("\n✓ Analysis complete!")
    
    if manual_count[0] > 0:
        print(f"\nMANUAL VERIFICATION:")
        print(f"  Your count: {manual_count[0]}")
        print(f"  Real-time diff: {abs(kickup_count - manual_count[0])}")
        print(f"  Post-analysis diff: {abs(len(peaks) - manual_count[0])}")
        
        if ground_truth_frames:
            print(f"  Ground truth diff: {abs(len(ground_truth_frames) - manual_count[0])}")
        
        if abs(kickup_count - manual_count[0]) < abs(len(peaks) - manual_count[0]):
            print(f"  ✓ Real-time is more accurate!")
        elif abs(len(peaks) - manual_count[0]) < abs(kickup_count - manual_count[0]):
            print(f"  ✓ Post-analysis is more accurate!")
        else:
            print(f"  ✓ Both equally accurate!")
    
except ImportError:
    print("matplotlib not available - skipping visualization")
except Exception as e:
    print(f"Visualization error: {e}")
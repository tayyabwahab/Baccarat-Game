# Baccarat Card Detection and Tracking System

A computer vision based intelligent system for detecting and tracking playing cards in Baccarat games using YOLOv5 object detection and custom centroid tracking algorithms. The project was designed for an edge device "Jetson nano"

## 🎯 Project Overview

This project implements an intelligent card detection and tracking system specifically designed for Baccarat table monitoring. It uses advanced computer vision techniques to:

- **Detect playing cards** in real-time using YOLOv5 deep learning model
- **Track individual cards** across video frames using centroid-based tracking
- **Classify card positions** as either "BANKER" or "PLAYER" based on their location
- **Assign unique IDs** to each detected card for consistent tracking
- **Provide real-time analysis** for live video streams or recorded footage

## ✨ Key Features

### 🃏 Card Detection
- **YOLOv5-based detection**: Utilizes state-of-the-art YOLOv5 model for accurate card detection
- **Real-time processing**: Optimized for live video streams with minimal latency
- **High confidence threshold**: Configurable detection confidence (default: 0.60)
- **Multi-card support**: Can detect and track multiple cards simultaneously

### 🎯 Object Tracking
- **Centroid-based tracking**: Custom implementation for robust card tracking
- **Unique ID assignment**: Each card gets a persistent ID across frames
- **Position classification**: Automatically categorizes cards as BANKER or PLAYER based on screen position
- **Disappearance handling**: Intelligently manages cards that temporarily leave the frame

### 📊 Real-time Analysis
- **Live visualization**: Real-time bounding boxes, IDs, and labels
- **Performance metrics**: Detailed timing information for each processing stage
- **Frame-by-frame logging**: Comprehensive tracking data for analysis
- **Webcam support**: Works with live camera feeds or video files

## 🛠️ Technical Architecture

### Core Components

1. **YOLOv5 Detection Engine** (`cards.py`)
   - Loads pre-trained YOLOv5 model (`model.pt`)
   - Processes video frames for card detection
   - Handles confidence thresholds and NMS (Non-Maximum Suppression)

2. **Centroid Tracker** (`CentroidTrackerFinal.py`)
   - Custom implementation for object tracking
   - Maintains object persistence across frames
   - Handles object registration/deregistration
   - Position-based classification (BANKER/PLAYER)

3. **Video Processing Pipeline**
   - Supports both webcam and video file input
   - Real-time frame processing and visualization
   - Configurable output options

### Model Details
- **Base Model**: YOLOv5 (You Only Look Once version 5)
- **Input Size**: 640x640 pixels (configurable)
- **Detection Classes**: Playing cards
- **Inference Speed**: Optimized for real-time processing

## 🚀 Deployment Platform

### Jetson Nano Deployment
This system has been successfully deployed and tested on **NVIDIA Jetson Nano**, providing:
- **Edge computing capabilities**: Local processing without cloud dependency
- **Optimized performance**: Leverages Jetson's GPU acceleration
- **Real-time processing**: Low-latency inference suitable for live applications
- **Power efficiency**: Optimized for embedded deployment scenarios
- **Compact form factor**: Ideal for table-side or overhead camera installations

## 📋 Requirements

### System Requirements
- Python 3.7+
- **Primary Platform**: NVIDIA Jetson Nano (tested and optimized)
- **Alternative**: CUDA-capable GPU (recommended for optimal performance)
- Webcam or video input source

### Python Dependencies
```
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0
tensorboard>=2.4.1
seaborn>=0.11.0
pandas
thop
```

## 🚀 Installation

### Jetson Nano Setup

1. **Prerequisites**
   ```bash
   # Update system packages
   sudo apt-get update && sudo apt-get upgrade
   
   # Install Python dependencies
   sudo apt-get install python3-pip python3-dev
   
   # Install OpenCV dependencies
   sudo apt-get install libopencv-dev python3-opencv
   ```

2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Baccarat-Game
   ```

3. **Install Python dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Jetson-specific optimizations**
   ```bash
   # Enable Jetson performance mode
   sudo nvpmodel -m 0
   sudo jetson_clocks
   
   # Set GPU memory allocation
   export CUDA_VISIBLE_DEVICES=0
   ```

5. **Verify installation**
   - Ensure `model.pt` is present in the project root
   - Test with webcam: `python3 cards.py`

### Alternative Platforms (PC/Laptop)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Baccarat-Game
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   - Ensure `model.pt` is present in the project root
   - Test with webcam: `python cards.py`

## 💻 Usage

### Basic Usage

#### Jetson Nano
```bash
python3 cards.py
```

#### PC/Laptop
```bash
python cards.py
```

This will start the system with default settings:
- **Input**: Webcam (device 0)
- **Confidence threshold**: 0.60
- **IOU threshold**: 0.45
- **Image size**: 640x640

### Configuration Options

The system can be configured by modifying parameters in `cards.py`:

```python
# Detection settings
conf_thres = 0.60      # Confidence threshold
iou_thres = 0.45       # NMS IOU threshold
max_det = 10           # Maximum detections per image

# Input settings
source = '0'           # '0' for webcam, or path to video file
imgsz = 640           # Input image size

# Output settings
view_img = False       # Show results window
save_txt = False       # Save results to text files
save_img = False       # Save processed images/videos
```

### Input Sources

1. **Webcam**
   ```python
   source = '0'  # Primary webcam
   source = '1'  # Secondary webcam
   ```

2. **Video File**
   ```python
   source = 'path/to/video.mp4'
   source = 'path/to/video.avi'
   ```

3. **Image Sequence**
   ```python
   source = 'path/to/images/'
   ```

## 📊 Output and Results

### Real-time Display
- **Bounding boxes**: Around detected cards
- **Object IDs**: Unique identifier for each tracked card
- **Position labels**: BANKER or PLAYER classification
- **Centroid markers**: Visual indicators of card centers

### Console Output
```
Frame 0
1 BANKER location = [320, 240]
2 PLAYER location = [960, 240]
Detection: (0.045s)
Tracking: (0.002s)
Total Time: (0.047s)
```

### Performance Metrics
- **Detection time**: YOLOv5 inference duration
- **Tracking time**: Centroid tracker processing time
- **Total processing time**: End-to-end frame processing

## 🔧 Advanced Configuration

### Model Customization
- **Model path**: Change `weights='model.pt'` to use different models
- **Class filtering**: Use `classes` parameter to filter specific card types
- **Half precision**: Enable `half=True` for faster inference (GPU only)

### Tracking Parameters
```python
# In CentroidTrackerFinal.py
maxDisappeared = 7  # Frames before deregistering object
```

### Visualization Options
```python
line_thickness = 3    # Bounding box thickness
hide_labels = False   # Show/hide class labels
hide_conf = False     # Show/hide confidence scores
```

## 🎮 Use Cases

### Casino Monitoring
- **Live table monitoring**: Real-time card tracking during games
- **Game analysis**: Post-game review and analysis
- **Security**: Detection of unusual card movements

### Research and Development
- **Computer vision research**: Object detection and tracking studies
- **Game theory analysis**: Card distribution pattern analysis
- **AI training**: Dataset generation for machine learning

### Educational Purposes
- **Computer vision learning**: Understanding object detection and tracking
- **Python programming**: Advanced image processing techniques
- **Real-time systems**: Performance optimization and latency management

## 🔍 Technical Details

### Detection Algorithm
- **YOLOv5**: Single-stage object detector
- **Anchor-based**: Pre-defined anchor boxes for different object sizes
- **Multi-scale**: Detects objects at different scales
- **NMS**: Non-Maximum Suppression for duplicate removal

### Tracking Algorithm
- **Centroid-based**: Uses object center points for tracking
- **Distance matching**: Euclidean distance for object association
- **Persistence**: Maintains object IDs across frames
- **Disappearance handling**: Automatic object deregistration

### Performance Optimization
- **GPU acceleration**: CUDA support for faster inference
- **Jetson optimization**: Leverages Tegra X1 GPU for edge computing
- **Batch processing**: Efficient handling of multiple detections
- **Memory management**: Optimized for real-time processing
- **Frame skipping**: Configurable processing frequency
- **Power management**: Jetson-specific power modes for optimal performance

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure `model.pt` is in the project root
   - Check file permissions and path

2. **Webcam not working**
   - Verify camera permissions
   - Try different device numbers (0, 1, 2...)
   - Check if camera is in use by other applications

3. **Poor detection accuracy**
   - Adjust confidence threshold (`conf_thres`)
   - Ensure proper lighting conditions
   - Check camera positioning and focus

4. **Performance issues**
   - Enable GPU acceleration if available
   - Reduce input image size (`imgsz`)
   - Use half-precision inference (`half=True`)

5. **Jetson Nano specific issues**
   - Ensure Jetson is in performance mode: `sudo nvpmodel -m 0`
   - Check GPU memory: `nvidia-smi`
   - Monitor temperature: `tegrastats`
   - Verify CUDA installation: `nvcc --version`

### Performance Tips
- **GPU usage**: Ensure CUDA is properly installed
- **Memory**: Monitor GPU memory usage for large models
- **Input quality**: Use high-quality video sources
- **Lighting**: Ensure adequate lighting for better detection

### Jetson Nano Optimization Tips
- **Power mode**: Use `sudo nvpmodel -m 0` for maximum performance
- **Fan control**: Monitor temperature with `tegrastats`
- **Memory management**: Use `sudo jetson_clocks` for optimal memory allocation
- **Thermal throttling**: Ensure proper cooling for sustained performance
- **Camera compatibility**: Use USB 3.0 cameras for better throughput

## 📈 Future Enhancements

### Planned Features
- **Card recognition**: Identify specific card values and suits
- **Game state analysis**: Automatic game outcome detection
- **Multi-table support**: Simultaneous monitoring of multiple tables
- **Cloud integration**: Remote monitoring capabilities
- **Mobile app**: Companion application for alerts and notifications

### Technical Improvements
- **Model optimization**: Quantization and pruning for edge devices
- **Jetson-specific optimizations**: TensorRT integration for faster inference
- **Multi-object tracking**: Enhanced tracking for complex scenarios
- **Temporal analysis**: Pattern recognition across multiple games
- **API integration**: RESTful API for external systems
- **Edge deployment**: Further optimization for embedded systems

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the code comments for implementation details

## 🙏 Acknowledgments

- **YOLOv5**: Ultralytics for the excellent object detection framework
- **OpenCV**: Computer vision library for image processing
- **PyTorch**: Deep learning framework
- **NVIDIA Jetson**: Edge computing platform for deployment
- **Computer Vision Community**: For research and development inspiration

---

**Note**: This system is designed for educational and research purposes. Please ensure compliance with local regulations when using in gaming environments. 
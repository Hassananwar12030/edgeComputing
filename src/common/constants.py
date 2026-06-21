"""
Project Constants
=================

Central location for all project-wide constants.
"""

# ===================
# Audio Processing
# ===================
AUDIO_SAMPLE_RATE = 16000  # 16 kHz
AUDIO_CHUNK_DURATION = 0.5  # 500ms
AUDIO_CHUNK_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION)  # 8000

# STFT parameters
STFT_N_FFT = 512
STFT_HOP_LENGTH = 160      # 10ms at 16kHz
STFT_WINDOW_LENGTH = 400   # 25ms at 16kHz
STFT_N_MELS = 128

# Spectrogram shape
SPECTROGRAM_SHAPE = (51, 128, 1)  # (time, freq, channels)

# ===================
# Vision Processing
# ===================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 10

# YOLOv8 classes of interest
CLASSES_OF_INTEREST = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "bird"
]

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
TEMPORAL_MIN_FRAMES = 3

# ===================
# Model Architecture
# ===================
NUM_CLASSES = 10
AUDIO_FEATURE_DIM = 128
VISION_FEATURE_DIM = 256
FUSION_DIM = AUDIO_FEATURE_DIM + VISION_FEATURE_DIM  # 384

# ===================
# Training
# ===================
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BUFFER_SIZE = 500
GRADIENT_ACCUMULATION_STEPS = 4

# ===================
# Communication
# ===================
# MQTT Topics
MQTT_TOPIC_DATA = "thesis/edge/{node_id}/data"
MQTT_TOPIC_WEIGHTS = "thesis/edge/{node_id}/weights"
MQTT_TOPIC_METRICS = "thesis/edge/{node_id}/metrics"
MQTT_TOPIC_MODEL = "thesis/server/model/global"
MQTT_TOPIC_COMMANDS = "thesis/server/commands"

# QoS levels
MQTT_QOS_METRICS = 0   # Best effort
MQTT_QOS_DATA = 1      # At least once
MQTT_QOS_WEIGHTS = 2   # Exactly once

# Default ports
MQTT_DEFAULT_PORT = 1883
FLOWER_DEFAULT_PORT = 8080

# ===================
# Federated Learning
# ===================
FL_DEFAULT_ROUNDS = 10
FL_MIN_CLIENTS = 2
FL_FRACTION_FIT = 1.0
FL_FRACTION_EVALUATE = 0.5

# ===================
# File Paths
# ===================
DEFAULT_CONFIG_EDGE = "configs/edge_config.yaml"
DEFAULT_CONFIG_SERVER = "configs/server_config.yaml"
DEFAULT_MODEL_PATH = "models/audio_cnn.h5"
DEFAULT_YOLO_PATH = "models/yolov8n.tflite"

# ===================
# System
# ===================
MAX_MEMORY_PERCENT = 70
LOG_MAX_SIZE_MB = 10
WATCHDOG_TIMEOUT = 60

# ===================
# Synchronization
# ===================
SYNC_TOLERANCE_MS = 50  # 50ms tolerance for A/V sync
SENSOR_BUFFER_SIZE = 30

# ===================
# Strategy Names
# ===================
STRATEGY_CENTRALIZED = "centralized"
STRATEGY_HYBRID = "hybrid"
STRATEGY_FEDERATED = "federated"

STRATEGIES = [
    STRATEGY_CENTRALIZED,
    STRATEGY_HYBRID,
    STRATEGY_FEDERATED
]

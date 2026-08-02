# System Architecture

## Overview

This document describes the complete system architecture for the federated learning
sensor fusion project. The system consists of three layers: Edge, Network, and Cloud.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLOUD LAYER (AWS/Laptop)                       │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Flower Server                               │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│   │  │   FedAvg    │  │  Strategy   │  │   Model Repository      │  │   │
│   │  │ Aggregator  │  │   Router    │  │   (Global Weights)      │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│   │                                                                  │   │
│   │  ┌─────────────────────────────────────────────────────────┐    │   │
│   │  │  Training Engine (Strategy A & B only)                  │    │   │
│   │  │  - Centralized Trainer                                  │    │   │
│   │  │  - Audio/Vision/Fusion Models                           │    │   │
│   │  └─────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                              MQTT (TLS)
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         NETWORK LAYER                                    │
│                    5 GHz Wi-Fi + MQTT Broker                            │
└─────────────────────────────────────────────────────────────────────────┘
                          │                    │
           ┌──────────────┘                    └──────────────┐
           ▼                                                  ▼
┌─────────────────────────────┐            ┌─────────────────────────────┐
│      EDGE NODE A            │            │      EDGE NODE B            │
│   (Raspberry Pi 4 - 8GB)    │            │   (Raspberry Pi 4 - 8GB)    │
│                             │            │                             │
│  ┌───────────┐ ┌─────────┐  │            │  ┌───────────┐ ┌─────────┐  │
│  │  Camera   │ │   Mic   │  │            │  │  Camera   │ │   Mic   │  │
│  └─────┬─────┘ └────┬────┘  │            │  └─────┬─────┘ └────┬────┘  │
│        ▼            ▼       │            │        ▼            ▼       │
│  ┌──────────────────────┐   │            │  ┌──────────────────────┐   │
│  │   Sensor Pipeline    │   │            │  │   Sensor Pipeline    │   │
│  └──────────┬───────────┘   │            │  └──────────┬───────────┘   │
│             ▼               │            │             ▼               │
│  ┌──────────────────────┐   │            │  ┌──────────────────────┐   │
│  │   Processing Layer   │   │            │  │   Processing Layer   │   │
│  │  YOLOv8 + STFT       │   │            │  │  YOLOv8 + STFT       │   │
│  └──────────┬───────────┘   │            │  └──────────┬───────────┘   │
│             ▼               │            │             ▼               │
│  ┌──────────────────────┐   │            │  ┌──────────────────────┐   │
│  │   Strategy Client    │   │            │  │   Strategy Client    │   │
│  └──────────────────────┘   │            │  └──────────────────────┘   │
└─────────────────────────────┘            └─────────────────────────────┘
```

---

## Component Details

### Edge Layer Components

#### 1. Sensor Pipeline (`src/edge/sensors/`)
- **camera.py**: Captures frames using picamera2
- **microphone.py**: Captures audio using I2S interface
- **sync.py**: Synchronizes audio-visual data using timestamps

#### 2. Processing Layer (`src/edge/processing/`)
- **stft.py**: Converts audio to mel-spectrograms
- **vision.py**: YOLOv8 inference for auto-labeling
- **fusion.py**: Feature extraction and fusion

#### 3. Strategy Layer (`src/edge/strategies/`)
- **base.py**: Abstract strategy interface
- **centralized.py**: Strategy A implementation
- **hybrid.py**: Strategy B implementation
- **federated.py**: Strategy C implementation

#### 4. Training Layer (`src/edge/training/`)
- **local_trainer.py**: On-device training for Strategy C
- **model.py**: TFLite model wrapper
- **dataset.py**: Local data buffer management

#### 5. Communication Layer (`src/edge/communication/`)
- **mqtt_client.py**: MQTT publish/subscribe
- **flower_client.py**: Flower FL client

#### 6. Metrics Layer (`src/edge/metrics/`)
- **power.py**: Power consumption (INA219)
- **bandwidth.py**: Network usage tracking
- **latency.py**: Inference timing

---

### Server Layer Components

#### 1. Flower Server (`src/server/flower_server.py`)
- Orchestrates FL rounds
- Manages client connections
- Handles model distribution

#### 2. Training Engine (`src/server/training/`)
- **centralized_trainer.py**: Training for Strategy A/B
- **models/**: Model architectures
  - `audio_cnn.py`: Audio classifier
  - `vision_cnn.py`: Vision model
  - `fusion_model.py`: Multimodal fusion

#### 3. Aggregation (`src/server/aggregation/`)
- **fedavg.py**: FedAvg algorithm implementation

#### 4. API (`src/server/api/`)
- **mqtt_handler.py**: MQTT message handling

---

## Data Flow

### Strategy A: Centralized

```
Edge:
  capture() → YOLOv8 → label
           → raw_audio + label → buffer

  when buffer full:
    → MQTT publish to server

Server:
  receive raw_audio, labels
  → STFT conversion
  → model.fit()
  → broadcast new model
```

### Strategy B: Hybrid STFT

```
Edge:
  capture() → YOLOv8 → label
           → STFT → spectrogram + label → buffer

  when buffer full:
    → MQTT publish to server

Server:
  receive spectrograms, labels
  → model.fit()
  → broadcast new model
```

### Strategy C: Federated

```
Edge:
  capture() → YOLOv8 → label
           → STFT → spectrogram + label → buffer

  when buffer full:
    → local model.fit()
    → Flower send weights to server

Server:
  receive weights from all clients
  → FedAvg aggregation
  → broadcast global model
```

---

## Open design note — fusion-era uploads for A/B (2026-08-02, not yet built)

The data flows above cover training the *audio* model. Training the full
*fusion* model server-side (Strategies A/B) also requires the vision input,
which the current flows never ship. Shipping raw frames would break the
privacy/bandwidth story. Proposed resolution, to revisit when fusion training
starts:

- Run frozen MobileNetV3 on the Pi; upload its 256-dim feature vector
  instead of the frame (small, not reversible to an image).
- Label still produced on-Pi by YOLO at capture time, as today.
- Per-sample upload becomes: (raw audio | spectrogram, vision_fv[256], label).
- Strategy C is unaffected (all inputs stay on-device).

---

## Model Architecture

### Fusion Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                FUSION MODEL (~1.33M params, ~390K trainable)         │
│                                                                      │
│  Vision Input (224×224×3)      Audio Input (51×128×1)               │
│         │                              │                             │
│         ▼                              ▼                             │
│  ┌─────────────────┐           ┌─────────────────┐                  │
│  │  MobileNetV3    │           │   Audio CNN     │                  │
│  │  (pre-trained)  │           │  (from scratch) │                  │
│  │  frozen layers  │           │  3× Conv2D+BN   │                  │
│  └────────┬────────┘           └────────┬────────┘                  │
│           │                             │                            │
│           ▼                             ▼                            │
│    256-dim features             128-dim features                     │
│           │                             │                            │
│           └──────────┬──────────────────┘                            │
│                      ▼                                               │
│              ┌─────────────────┐                                     │
│              │  Concatenate    │                                     │
│              │    (384-dim)    │                                     │
│              └────────┬────────┘                                     │
│                       ▼                                              │
│              ┌─────────────────┐                                     │
│              │  Fusion Head    │                                     │
│              │  Dense(256)     │                                     │
│              │  Dropout(0.3)   │                                     │
│              │  Dense(N_class) │                                     │
│              └────────┬────────┘                                     │
│                       ▼                                              │
│                  Prediction                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Communication Protocol

### MQTT Topics

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `thesis/edge/{id}/data` | Edge → Server | Raw data/spectrograms (A/B) |
| `thesis/edge/{id}/weights` | Edge → Server | Model weights (C) |
| `thesis/edge/{id}/metrics` | Edge → Server | Performance metrics |
| `thesis/server/model/global` | Server → Edge | Global model updates |
| `thesis/server/commands` | Server → Edge | Control messages |

### QoS Levels

- Model weights: QoS 2 (exactly once)
- Data upload: QoS 1 (at least once)
- Metrics: QoS 0 (best effort)

---

## Security

- TLS 1.3 encryption for all MQTT communication
- Client certificates for each edge node
- Message signing for model weights

---

## Resource Constraints

### Raspberry Pi 4 (8GB)

| Resource | Constraint | Mitigation |
|----------|------------|------------|
| RAM | 8GB | Batch size 8, gradient accumulation |
| CPU | 4 cores | Async processing, prioritize inference |
| Storage | 32GB SD | Rotate logs, compress data |
| Power | 5W | Event-based capture, sleep modes |

### Network

| Scenario | Bandwidth | Mitigation |
|----------|-----------|------------|
| Strategy A | 4MB/batch | Event-based upload |
| Strategy B | 3MB/batch | Spectrogram compression |
| Strategy C | 2MB/round | Weight quantization |

---

## Scalability Considerations

- System designed for 2 edge nodes (thesis scope)
- Architecture supports N nodes with minimal changes
- FedAvg scales linearly with clients
- MQTT broker can handle hundreds of clients

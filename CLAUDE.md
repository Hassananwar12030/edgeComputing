# CLAUDE.md - AI Assistant Context File

> **Purpose:** This file provides context for Claude (or other AI assistants) to understand
> this project quickly in future sessions. Read this first before helping with the project.

---

## Project Summary (TL;DR)

This is a **Master's thesis** implementing **federated learning** for **audio-visual sensor fusion** on **Raspberry Pi** devices. The goal is to train a model that recognizes objects by their **sound** (so robots can "see" in fog/darkness).

**Key Point:** There are TWO models:
1. **YOLOv8** (fixed, pre-trained) - Looks at camera, provides labels ("car", "person")
2. **Fusion Model** (being trained) - Learns to recognize objects from audio spectrograms

---

## Critical Concepts to Understand

### 1. The Labeling Assumption
Audio samples are labeled automatically by YOLOv8 vision detection:
- Camera sees car → YOLOv8 says "car" → microphone sound labeled as "car"
- This is "weak supervision" - labels are noisy but work with enough data

### 2. Three Strategies Being Compared

| Strategy | What happens on Pi | What goes to server | Server does |
|----------|-------------------|--------------------:|-------------|
| A: Centralized | YOLOv8 only | Raw audio + labels | Full training |
| B: Hybrid | YOLOv8 + STFT | Spectrograms + labels | Full training |
| C: Federated | YOLOv8 + STFT + Train | Model weights only | FedAvg (averaging) |

### 3. Model Freshness
How quickly model adapts to new objects:
- Centralized (A/B): 12-48 hours (slow)
- Federated (C): 5-30 minutes (fast)

### 4. Training is BATCHED, not real-time
- Collect 500 samples locally
- Then trigger training/upload
- NOT every frame!

---

## Project Structure Quick Reference

```
src/edge/           # Runs on Raspberry Pi
├── sensors/        # Camera + microphone capture
├── processing/     # STFT, YOLOv8 inference
├── strategies/     # A, B, C implementations
├── training/       # Local training (Strategy C only)
├── communication/  # MQTT + Flower client
└── metrics/        # Power, bandwidth measurement

src/server/         # Runs on laptop/cloud
├── training/       # Centralized training (A/B)
├── aggregation/    # FedAvg (C)
└── api/            # MQTT handler

src/common/         # Shared code
configs/            # YAML configurations
models/             # Pre-trained weights
```

---

## Key Technical Details

### Audio Processing
- Sample rate: 16,000 Hz
- Window: 25ms, Hop: 10ms
- FFT size: 512
- Mel bins: 128
- Output: (49, 128, 1) spectrogram for 500ms audio

### Fusion Model Architecture
```
Vision Branch:    MobileNetV3-Small (pre-trained) → 256-dim
Audio Branch:     3× Conv2D + BatchNorm + MaxPool → 128-dim
Fusion Head:      Concat(384) → Dense(256) → Dropout → Dense(N_classes)
Total:            ~2M parameters
```

### Communication
- Protocol: MQTT with TLS
- FL Framework: Flower (flwr)
- Topics: `thesis/edge/{node_id}/data`, `thesis/server/model/global`

---

## Common Tasks & How to Help

### "Help me implement X strategy"
→ Check `src/edge/strategies/` for the strategy pattern
→ Each strategy implements `ILearningStrategy` interface

### "Help with audio processing"
→ Check `src/edge/processing/stft.py`
→ Uses librosa or numpy for STFT

### "Help with the fusion model"
→ Check `src/server/training/models/fusion_model.py`
→ TensorFlow/Keras implementation

### "Help with federated learning"
→ Check `src/edge/communication/flower_client.py`
→ Uses Flower framework

### "Why isn't model training?"
→ Check if buffer has 500+ samples
→ Check if training trigger conditions are met
→ Check MQTT connection to server

---

## Implementation Status

Check `docs/IMPLEMENTATION_STATUS.md` for current progress.

### What's Done
- [x] Project structure
- [x] Documentation
- [ ] Sensor integration (TODO)
- [ ] STFT pipeline (TODO)
- [ ] YOLOv8 integration (TODO)
- [ ] Strategy implementations (TODO)
- [ ] Flower integration (TODO)
- [ ] Evaluation framework (TODO)

---

## Configuration Files

### `configs/edge_config.yaml`
- Node ID, server address
- Camera/microphone settings
- Buffer sizes, training triggers

### `configs/server_config.yaml`
- FL rounds, aggregation strategy
- Model hyperparameters

### `configs/strategies.yaml`
- Strategy-specific parameters

---

## Testing Approach

```bash
# Run all tests
pytest tests/

# Run edge tests only
pytest tests/edge/

# Run specific test
pytest tests/edge/test_stft.py -v
```

---

## Debugging Tips

### On Raspberry Pi
- Check camera: `libcamera-hello`
- Check I2S mic: `arecord -l`
- Check MQTT: `mosquitto_sub -t "thesis/#" -v`

### On Server
- Check Flower server logs
- Check MQTT broker: `mosquitto -v`

---

## Important Constraints

1. **Raspberry Pi 4 (8GB)** - Limited resources
   - Use batch size 8, gradient accumulation
   - Use TFLite for inference
   - Swap enabled (2GB)

2. **AWS Free Tier** - Limited compute
   - Strategy C (federated) works fine
   - Strategy A/B may be slow → use laptop instead

3. **Privacy** - Core thesis concern
   - Strategy C: Data never leaves device
   - Strategy A: Raw audio leaves device (privacy risk)

---

## Questions This Project Answers

1. How does federated learning compare to centralized for sensor fusion?
2. What are the privacy/power/accuracy trade-offs?
3. Can audio help when vision fails (fog, darkness)?
4. How fast can each strategy adapt to new objects?

---

## Files to Read First (in order)

1. This file (`CLAUDE.md`)
2. `README.md` - Project overview
3. `docs/ARCHITECTURE.md` - System design
4. `configs/edge_config.yaml` - Configuration
5. `src/edge/main.py` - Edge entry point
6. `src/server/flower_server.py` - Server entry point

---

## Contact / Supervision

- **Author:** Adnan Anwar Rajput
- **First Supervisor:** Prof. Dr. André Jakob
- **Second Supervisor:** Prof. Dr. Marcus Purat
- **Institution:** Berliner Hochschule Für Technik, Department VII

---

*Last updated: 2026*

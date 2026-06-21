# Implementation Status

> **Last Updated:** 2026-06-21
> **Current Phase:** Bootstrap complete → Strategy comparison (A/B/C) next
> **Phase 2 (Pi sensor integration):** deferred until hardware is ready

---

## Where we are (TL;DR)

The audio classification pipeline works end-to-end on real data.

- **YOLO labeler** (the auto-labeling oracle): verified, ~80% precision when it fires.
- **STFT pipeline**: WAV → 16 kHz → 500ms center crop → mel-spectrogram `(51, 128)`.
- **Audio classifier (AudioCNN)**: 111K params, trained on UrbanSound8K, **68.6% test accuracy** on held-out fold 10.
- **Fusion model (audio + vision)**: builds and forward-passes (1.33M params, MobileNetV3 frozen + audio CNN + fusion head).
- All re-runnable from `scripts/` with a single command each.

What's not done yet: the **actual thesis content** — implementing and comparing the three learning strategies (Centralized, Hybrid, Federated). That's what's next.

---

## Overall Progress

```
[██████░░░░░░░░░░░░░░] 30% Complete
```

---

## Phase Checklist

### Phase 1: Project Setup ✅
- [x] Project structure + documentation
- [x] Virtual environment (`.venv`, Python 3.13)
- [x] Dependencies (TensorFlow 2.21, PyTorch 2.12 via ultralytics, librosa, etc.)
- [x] Scaffolded code reviewed, several bugs fixed (see "Bugs found")

### Phase 2: Sensor Integration ⏸ DEFERRED
- [ ] Camera capture (picamera2)
- [ ] Microphone capture (I2S)
- [ ] Audio-visual synchronization
- [ ] Test on Raspberry Pi hardware

**Why deferred:** Hardware not yet wired up. We can build and validate the
strategy comparison machinery on mock data (UrbanSound8K replay) without
sensors, since the thesis is about *relative* performance of the three
strategies, not their absolute accuracy.

### Phase 3: Audio Processing Pipeline ✅
- [x] STFT implementation (`src/edge/processing/stft.py`)
- [x] Mel-spectrogram generation: `(51, 128, 1)` output
- [x] Visual sanity check: spectrograms are class-distinguishable per UrbanSound8K class
- [x] Fixed `(49, 128)` → `(51, 128)` scaffold inconsistency across 8 files

### Phase 4: Vision Pipeline ✅ (laptop only)
- [x] YOLOv8n loaded and verified (`scripts/test_yolo.py`)
- [x] Inference latency: ~32 ms / image on M-series Mac
- [x] Auto-labeling assumption validated on VGGSound (~80% precision when YOLO labels)
- [ ] TFLite conversion for Raspberry Pi (deferred with Phase 2)

### Phase 5: Communication Layer ❌
- [ ] MQTT broker (mosquitto)
- [ ] Flower server
- [ ] TLS / certificates
- [ ] Serialization formats

### Phase 6: Strategy A (Centralized) ❌
### Phase 7: Strategy B (Hybrid) ❌
### Phase 8: Strategy C (Federated) ❌

(Strategy scaffolds exist in `src/edge/strategies/` but are not yet wired to
real data or comm layer.)

### Phase 9: Fusion Model 🟡 (scaffold + bootstrap)
- [x] AudioCNN architecture (111K params)
- [x] AudioCNN trained on UrbanSound8K — **68.6% test accuracy**
- [x] FusionModel architecture (1.33M params)
- [x] FusionModel forward pass verified
- [ ] Modality dropout
- [ ] Train fusion model on paired audio+video data (requires Phase 2 or VGGSound pull)
- [ ] TFLite conversion

### Phase 10: Metrics & Evaluation ❌
### Phase 11: Experiments ❌
### Phase 12: Thesis Writing ❌

---

## What's been done — narrative

1. **Bootstrap step 1 — YOLO labeler verified.** Standard YOLOv8n produces correct
   labels on 5 ground-level test images. Latency ~32 ms on laptop.

2. **Bootstrap step 2 — Labeling assumption validated.** Pulled 150 thumbnails
   from VGGSound (5 classes × 30 clips), ran YOLO on each, compared against
   the human audio label. Result: **~80% precision** when YOLO fires
   (recall lower, but no-detection samples are simply skipped — not noise).

3. **Bootstrap step 3 — STFT pipeline verified.** Each UrbanSound8K WAV gets
   loaded, resampled to 16 kHz, center-cropped to 500ms, transformed to a
   (51, 128) mel-spectrogram. Spectrograms visually distinguishable per class.

4. **Bootstrap step 4 — Audio classifier trained.** Two passes:
   - v1: center-crop only → 66.3% test accuracy, undertrained.
   - v2: sliding-window augmentation (4 windows/clip) + early stopping →
     **68.6% test accuracy** (target was 70-75%, missed; accepted as baseline).

---

## Bugs found in scaffolds (fixed)

| File | Bug |
|------|-----|
| `src/edge/processing/vision.py` | `classes_of_interest=None` didn't disable filter as docstring claimed |
| `src/edge/processing/stft.py` + 7 other files | Hardcoded `(49, 128)` shape inconsistent with librosa default `center=True` (real output is `(51, 128)`) |
| `src/server/__init__.py` | Eagerly imported `FlowerServer`, breaking on machines without Flower installed |

---

## Known limitations (flag in thesis writeup)

1. **AudioCNN is 111K params**, not the "~500K" estimated in `CLAUDE.md` /
   `ARCHITECTURE.md`. Either update docs, or upsize the model before final
   experiments.
2. **Single-fold split** (fold 10 holdout) reported instead of 10-fold CV.
   Run 10-fold (~5 hours) before the thesis report for proper `mean ± std`.
3. **`siren` class accuracy** is the weakest link (44% F1). If thesis demos
   highlight emergency-vehicle / safety scenarios, this needs work.
4. **Latent bug:** `src/server/flower_server.py` references undefined `Metrics`
   type in an annotation. Will trip when Strategy C work starts — fix is
   one line (`from __future__ import annotations`).

---

## What's next — implementation plan

### Step 5: In-process strategy harness (no hardware, no real network)
- Replay UrbanSound8K as a stream of "incoming sensor data"
- Implement Strategy A/B/C as in-process Python (no MQTT/Flower yet)
- Bandwidth = bytes that *would* be transmitted (serialize but don't send)
- Output: side-by-side comparison table (accuracy, simulated bandwidth, time)

### Step 6: Realistic communication
- Run mosquitto MQTT broker locally
- Strategy A/B uses real MQTT publish/subscribe
- Strategy C uses real Flower server + clients on localhost
- Now bandwidth and latency are real

### Step 7: Model freshness experiment (the central thesis demo)
- Pre-train on 8 of 10 classes
- Start runtime, introduce 2 unseen classes mid-stream
- Measure time-to-first-correct-prediction for each strategy
- This directly tests the thesis hypothesis: federated learning adapts faster

### Step 8: Non-IID experiment
- Distribute classes unevenly across simulated edge nodes
- Run Strategy C with vanilla FedAvg
- Show client drift effect and (optionally) FedProx mitigation

### Then — Phase 2: Sensor integration (when Pi hardware ready)
- Camera capture (picamera2)
- I2S mic capture
- Synchronized capture loop
- Re-run steps 5-8 on real sensor data

### Then — Phase 12: Thesis writeup
- Results analysis
- Figures, tables, comparisons
- Final document

---

## Recent Changes

| Date | Change |
|------|--------|
| 2026-05-31 | Initial project structure |
| 2026-06-20 | YOLO labeler verified end-to-end |
| 2026-06-20 | Label-noise measurement on VGGSound: ~80% precision |
| 2026-06-21 | UrbanSound8K downloaded, STFT pipeline verified |
| 2026-06-21 | AudioCNN + FusionModel build & forward-pass verified |
| 2026-06-21 | AudioCNN trained on UrbanSound8K: **68.6% test accuracy** |
| 2026-06-21 | Phase 2 (sensors) deferred — strategy work begins next |

---

## Hardware Status

| Device | Status | Notes |
|--------|--------|-------|
| Raspberry Pi A | Not in use | Will integrate in Phase 2 (later) |
| Raspberry Pi B | Not in use | |
| Camera Module 3 | Not yet wired | |
| I2S Microphone | Not yet wired | |
| 5GHz Router | n/a yet | |

---

## Re-runnable scripts (current state)

```bash
# Bootstrap pipeline (all already executed; re-run any time)
.venv/bin/python scripts/test_yolo.py
.venv/bin/python scripts/fetch_vggsound_thumbnails.py
.venv/bin/python scripts/measure_label_noise.py
.venv/bin/python scripts/test_stft.py
.venv/bin/python scripts/test_audio_cnn.py
.venv/bin/python scripts/test_fusion_model.py
.venv/bin/python scripts/prepare_urbansound8k.py
.venv/bin/python scripts/train_audio_cnn.py
```

---

## Open questions (need decision before Step 5)

1. Mock-data design: replay UrbanSound8K straight, or hold out classes
   for the model-freshness experiment (Step 7)?
2. Number of simulated edge nodes: 2 (thesis-stated minimum) or N for
   robustness curves?
3. When do the Pis arrive / get wired? (Affects whether to invest in better
   mocks or wait for real hardware.)

# Fusion — the sequel to the Primer

[PRIMER.md](PRIMER.md) ends with "the fusion model is deferred." It isn't
anymore: this doc teaches, in the same from-zero style, what we built, how it
failed first, how we diagnosed and fixed it, what the results prove, how to
get the data yourself, and what remains (F6/F7).

Read PRIMER first — this assumes its concepts (model, training, weights,
spectrogram, weak supervision, epoch).

---

# Part 1 — Why fusion, and why it needed new data

## 1.1 What the fusion model is
Until now the trained model used **audio only**. The **FusionModel** uses two
senses at once: a **vision branch** (MobileNetV3 — a small, pre-trained,
frozen image network) and the **audio branch** (the same CNN architecture as
our AudioCNN), whose outputs are merged ("fused") into one prediction.
Why bother: to directly demonstrate the thesis's motivating claim — **"audio
helps when vision fails."** A fused model should work best with both senses
and *keep working* when the camera dies.

## 1.2 Why UrbanSound8K couldn't train it
Training a two-sense model needs examples with BOTH senses recorded at the
same moment, plus a label: **(image, audio, label)** triples. UrbanSound8K is
audio-only — no images. So we needed a *paired* dataset.

## 1.3 Where the pairs came from: VGGSound
**VGGSound** is an Oxford dataset: ~200,000 ten-second YouTube clips, each
with a human label for the *sound* ("dog barking"). Crucially, Oxford doesn't
distribute the videos (copyright) — only a CSV of YouTube IDs. Everyone
re-downloads the clips themselves. We did the same:

- For 5 classes (car, motorcycle, dog, cat, bird) we downloaded ~200 clips
  each, and from every clip extracted **the middle frame** (what the camera
  saw) and **the audio** (16 kHz mono). Result: **973 (image, audio, label)
  pairs**.
- Labels here are the **human** VGGSound labels — YOLO is not needed in this
  loop. (YOLO remains the labeler for *live Pi* capture; VGGSound gives us
  clean labels for training the fusion model. Two different label sources for
  two different jobs.)

## 1.4 How YOU get the data (two ways)

**Way 1 — fastest (recommended):** get `paired_cache.npz` (~164 MB) from the
shared Drive link, drop it at `data/vggsound/paired_cache.npz`. Done — every
script below runs. (It is not in git: GitHub blocks files >100 MB, and the
raw media is YouTube-derived content we shouldn't redistribute publicly.)

**Way 2 — regenerate it yourself (~90 min):**
```bash
# 1. the official Oxford CSV of YouTube IDs
mkdir -p data/vggsound
curl -L -o data/vggsound/vggsound.csv \
    https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv

# 2. harvest pairs (yt-dlp + ffmpeg needed; resumable — rerun to continue)
.venv/bin/pip install yt-dlp   # and: brew/apt install ffmpeg
.venv/bin/python scripts/spike_vggsound_pairs.py --per-class 200 \
    --max-attempts-per-class 700

# 3. build the training cache
.venv/bin/python scripts/prepare_vggsound_pairs.py
```
Expect ~40% of download attempts to fail (deleted/private videos — "link
rot"); the script just tries more IDs. Your copy will differ slightly from
ours (rot moves) — that's normal for VGGSound-based work.

**What the cache contains:** each 10 s clip became one 224×224 frame + several
500 ms audio windows → spectrograms. 3,008 training samples from 752 clips,
plus val (92) and test (98). The split is **by clip** — all windows of one
clip stay in one split, otherwise the model would be tested on near-copies of
its training data (leakage).

---

# Part 2 — What we trained, and the failure that taught us the most

## 2.1 Three models, not one
We trained the FusionModel **and two baselines on the same data**:
- **audio-only** — just the audio branch + classifier
- **vision-only** — just MobileNetV3 + classifier

Why: "fusion scored 85%" means nothing alone. The claim "fusion helps" needs
*fusion > audio-only AND fusion > vision-only* on identical data. Baselines
are the ruler you measure with.

## 2.2 First attempt: naive fusion — and it failed
Results of the first round: vision-only 0.806, **fusion 0.745**, audio-only
0.561. Fusion was *worse* than vision alone. And a harsher test made it
obvious why (see 2.4): the fused model was **completely ignoring audio**.

## 2.3 Why: modality gradient starvation (the concept)
The vision branch is *pre-trained* — it produces useful features from the
first minute. The audio branch starts *from random* — early on its output is
noise. During training, the model finds it can explain the labels using
vision alone, the loss drops, and the gradients (the "learning signal")
flowing to the audio branch shrink to almost nothing. The audio branch never
grows up; the fusion head learns to wire it out. A strong sense starves the
weak one. This is a known, named failure mode of multimodal training.

## 2.4 How we caught it (debugging you can reuse)
Three probes, each cheap:
1. **Suspiciously identical accuracies** across supposedly-different runs —
   an "identical number" is a bug smell, not a coincidence.
2. **Blackout prediction check:** feed black images + real audio → the model
   predicted ONE constant class for all 98 test clips, and swapping the audio
   changed 0/20 predictions. Audio influence: zero.
3. **Layer probe:** the audio branch's 128-number output barely changed
   between different sounds (57% dead zeros). The branch itself was flat.
Lesson: when a result looks wrong, probe *inside* the model instead of
re-running training blindly.

## 2.5 The two-part fix
- **Audio warm-start:** copy the weights of the already-trained audio-only
  model into the fusion model's audio branch before training. Now audio
  features are informative from step one, and the head has a reason to use
  them. (Same philosophy as warm-starting the strategies from the trained
  AudioCNN.)
- **Modality dropout (50%):** during training, half the samples get their
  image replaced by black. The model cannot always lean on vision, so it
  must keep an audio pathway alive. (Dropout *alone* did not work — it can't
  revive a starved branch; it only maintains one that already functions.
  Order matters: warm-start first, dropout second.)

---

# Part 3 — The results, explained

## 3.1 The blackout experiment
Evaluate every model twice on the same 98 test clips:
- **clean** — real frames + audio (a sunny day)
- **blackout** — every frame black, audio unchanged (fog / darkness / dead
  camera)

## 3.2 The table

| Model | Clean | Blackout | What it means |
|---|---|---|---|
| vision-only | 0.806 | 0.204 | blind in the dark (0.20 = random guessing) |
| naive fusion | 0.745 | 0.173 | ignored audio → dies with vision |
| dropout only | 0.745 | ~0.21 | starved branch — dropout can't save it |
| **warm-start + dropout** | **0.847** | **0.551** | **best on clean AND survives the dark** |
| audio-only | 0.561 | 0.561 | the "camera dead" floor |

## 3.3 The two claims this proves
1. **Fusion beats either sense alone:** 0.847 > 0.806 (vision) > 0.561
   (audio). Two senses genuinely combine — with the right recipe.
2. **Audio helps when vision fails** — the thesis's motivating sentence,
   now measured: in darkness, vision-only collapses to random (0.204) while
   the fused model keeps 0.551, essentially everything audio alone can give
   (0.561). Graceful degradation instead of collapse.

## 3.4 Honesty notes
- The test set is small (98 clips) → read each number as ±~5%.
- The *recipe* (warm-start, then dropout) is itself a finding: naive
  multimodal training silently fails, and the failure is invisible in clean
  accuracy alone — you must test the failure condition (blackout).
- VGGSound clips are videos OF the labeled object → vision is unusually
  strong here. Real street scenes would be kinder to audio.

---

# Part 4 — What remains (yours to run)

## 4.1 F6 — strategies A/B/C with the fusion model
So far the strategy comparison (see PRIMER Part 5) trained the audio-only
AudioCNN. F6 re-runs it with the FusionModel in the training slot. One design
decision is already made (see ARCHITECTURE.md §2 note): in A/B the edge ships
the MobileNetV3 **feature vector** (256 numbers) instead of the raw frame —
smaller, and not reversible into an image, so the privacy story holds.
Strategy C is unchanged (everything stays on-device). Expected outcome: same
trade-off shape as before (bandwidth ↓, edge compute ↑ as training moves to
the edge), now with the bigger model.

## 4.2 F7 — Pi feasibility
Measure on the real Pi 5: MobileNetV3 feature extraction per frame, and one
epoch of fusion training on a small buffer. Answers "can the fusion-era edge
actually run on the hardware?" — a table of timings, no new concepts.

## 4.3 Command cheat-sheet
```bash
# train everything from scratch (baselines + naive fusion)
.venv/bin/python scripts/train_fusion.py --model all

# the working recipe (the headline model)
.venv/bin/python scripts/train_fusion.py --model fusion \
    --modality-dropout 0.5 --epochs 25 --es-patience 25 --audio-warmstart

# the blackout evaluation (prints the table from Part 3)
.venv/bin/python scripts/eval_fusion_blackout.py
```
Outputs: `models/fusion_*.keras`, `data/vggsound/fusion_training_metrics.json`,
`data/vggsound/fusion_blackout_results.json`.

---

# Glossary additions (beyond PRIMER's)

- **Paired data** — samples containing both modalities from the same moment:
  (image, audio, label).
- **Link rot** — YouTube videos in the VGGSound list that no longer exist.
- **Leakage** — test data that overlaps training data (e.g. windows of the
  same clip in both) → inflated, meaningless accuracy. Prevented by splitting
  by clip.
- **Modality gradient starvation** — a pre-trained branch soaks up the
  learning signal; the from-scratch branch never develops and gets ignored.
- **Warm-start** — initialize part of a model from an already-trained model
  instead of random.
- **Modality dropout** — randomly blank one modality during training to force
  competence in the other.
- **Blackout evaluation** — test with one modality disabled to measure
  graceful degradation (our "fog test").

# Fusion — the sequel to the Primer

[PRIMER.md](PRIMER.md) ends with "the fusion model is deferred." It isn't
anymore. This doc explains how the fusion model works, what we did step by
step, what the results prove, how to get the data, and what remains (F6/F7).

Read PRIMER first — this assumes its concepts (model, training, weights,
spectrogram, epoch).

---

# Part 1 — How the fusion model works

## 1.1 The idea in one sentence
One model, two senses: it looks at a camera frame **and** listens to half a
second of audio, and combines both into a single prediction — so that when
one sense fails (camera in darkness), the other still carries it.

## 1.2 The architecture, piece by piece

```
  camera frame                          500 ms audio
  (224 × 224 RGB)                       (waveform)
        │                                    │
        │                                  STFT
        │                                    │
        ▼                                    ▼
┌─────────────────┐               ┌──────────────────────┐
│ VISION BRANCH   │               │ AUDIO BRANCH         │
│ MobileNetV3-    │               │ 3× Conv blocks       │
│ Small (frozen,  │               │ (same architecture   │
│ pre-trained)    │               │ as our AudioCNN)     │
└────────┬────────┘               └──────────┬───────────┘
         ▼                                   ▼
  vision feature vector              audio feature vector
      (256 numbers)                     (128 numbers)
         └───────────────┬───────────────────┘
                         ▼
              concatenate → 384 numbers
                         ▼
              ┌─────────────────────┐
              │ FUSION HEAD         │
              │ Dense(256) → 128    │
              │ → softmax           │
              └──────────┬──────────┘
                         ▼
        prediction: [car 0.81, dog 0.05, ...]
```

**Vision branch.** MobileNetV3-Small, a small image network pre-trained on
millions of photos, **frozen** (we never train it). Give it a frame, it
outputs a **feature vector**: 256 numbers that summarize *what's visible* —
not a label, a compact description. Think of it as the model's eye.

**Audio branch.** The same CNN architecture as our AudioCNN. The audio is
first turned into a spectrogram (PRIMER §11), and the branch compresses it
to a 128-number feature vector — a summary of *what's audible*. The ear.
Unlike the eye, this branch **is trained** — hearing these objects is exactly
the skill nobody ships pre-trained.

**Fusion head.** The two vectors are **concatenated** (glued end to end:
256 + 128 = 384 numbers) and passed through two small dense layers ending in
a softmax that outputs a probability per class. This head is where "judgment"
lives: it learns *which* numbers from *which* sense matter for each class —
e.g. lean on the eye for "car", on the ear when the eye's numbers look like
darkness. This style — each sense encoded separately, merged at the feature
level — is called **late fusion**.

**What trains, what doesn't:** vision branch frozen; audio branch + fusion
head learn (~390K of the model's ~1.33M weights).

## 1.3 Why this survives darkness (when trained right)
In darkness the eye still outputs 256 numbers — but they're the *same-ish*
numbers for every black frame, carrying no information. If the head has
learned real audio pathways, the ear's 128 numbers still separate the
classes, and the prediction survives. Making sure the head actually learns
those audio pathways is the crux of Part 3.

---

# Part 2 — The data: paired examples

## 2.1 Why new data was needed
Training a two-sense model needs examples with both senses captured at the
same moment plus a label: **(image, audio, label)**. UrbanSound8K is
audio-only, so it can't train the vision side.

## 2.2 VGGSound, and how we harvested it
**VGGSound** (Oxford) is ~200,000 ten-second YouTube clips, each with a human
label for the sound ("dog barking"). Oxford distributes only a CSV of YouTube
IDs (the videos are copyrighted) — everyone downloads the clips themselves.

We did exactly that for 5 classes (**car, motorcycle, dog, cat, bird**):
download ~200 clips per class, extract from each the **middle frame** + the
**audio** (16 kHz mono) → **973 (image, audio, label) pairs**. Labels are the
human VGGSound labels; YOLO is not involved here. (YOLO stays the labeler for
*live Pi* capture — a different job.)

The pairs are then packed into a training cache: each clip contributes one
frame + several 500 ms audio windows (→ spectrograms). 3,008 training
samples from 752 clips; validation 92, test 98. The split is **by clip** —
all windows of a clip stay in one split, or the model would be tested on
near-copies of its training data (**leakage**).

## 2.3 How YOU get the data

**Fastest:** take `paired_cache.npz` (~164 MB) from the shared Drive link and
place it at `data/vggsound/paired_cache.npz`. Everything below then runs.
(It's not in git — GitHub blocks >100 MB files, and the media is
YouTube-derived content we shouldn't republish.)

**Or regenerate (~90 min, resumable):**
```bash
mkdir -p data/vggsound
curl -L -o data/vggsound/vggsound.csv \
    https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv
.venv/bin/pip install yt-dlp        # plus: brew/apt install ffmpeg
.venv/bin/python scripts/spike_vggsound_pairs.py --per-class 200 \
    --max-attempts-per-class 700
.venv/bin/python scripts/prepare_vggsound_pairs.py
```
~40% of attempts fail (deleted/private videos — "link rot"); the script just
tries more IDs. Your copy will differ slightly from ours — normal for
VGGSound work.

---

# Part 3 — What we did, step by step

**Step 1 — Train three models on the same data.** The FusionModel *and* two
baselines: **audio-only** (ear + classifier) and **vision-only** (eye +
classifier). Baselines are the ruler: "fusion helps" is only claimable as
*fusion > audio-only AND fusion > vision-only* on identical data.

**Step 2 — Warm-start the ear.** Before training the fusion model, we copy
the weights of the already-trained audio-only model into the fusion model's
audio branch. Reason (short version): a from-scratch ear next to a
pre-trained eye never gets a chance — the model solves training using vision
alone and the ear stays undeveloped. Starting the ear already-competent makes
the head take it seriously. (The full failure-and-diagnosis story: Appendix.)

**Step 3 — Train with modality dropout.** During training, **half the
samples get their image replaced by black**. The model can't always lean on
its eye, so it must keep the audio pathways alive. This is what buys graceful
degradation in darkness. Together, steps 2+3 are "the recipe":

```bash
.venv/bin/python scripts/train_fusion.py --model fusion \
    --modality-dropout 0.5 --epochs 25 --es-patience 25 --audio-warmstart
```

**Step 4 — The blackout evaluation ("fog test").** Every model is evaluated
twice on the same 98 test clips: **clean** (real frames + audio) and
**blackout** (every frame black, audio unchanged — simulating fog, darkness,
or a dead camera):

```bash
.venv/bin/python scripts/eval_fusion_blackout.py
```

---

# Part 4 — The results, explained

| Model | Clean | Blackout | What it means |
|---|---|---|---|
| vision-only | 0.806 | 0.204 | blind in the dark (0.20 = random guessing) |
| naive fusion (no recipe) | 0.745 | 0.173 | learned to ignore audio → dies with vision |
| **fusion with recipe** | **0.847** | **0.551** | **best on clean AND survives the dark** |
| audio-only | 0.561 | 0.561 | the "camera dead" floor |

**Claim 1 — fusion beats either sense alone:** 0.847 > 0.806 (vision) >
0.561 (audio). The two senses genuinely combine — but only with the recipe;
naive fusion was *worse* than vision alone.

**Claim 2 — audio helps when vision fails** (the thesis's motivating
sentence, now measured): in blackout, vision-only collapses to random
(0.204) while the fused model keeps 0.551 — essentially everything audio
alone can give (0.561). Graceful degradation instead of collapse.

**Reading honestly:** the test set is small (98 clips), so treat each number
as ±~5%. And VGGSound frames are unusually vision-friendly (the clips are
videos *of* the labeled object), which flatters the vision baseline — real
street scenes would be kinder to audio.

---

# Part 5 — What remains (yours to run)

**F6 — strategies A/B/C with the fusion model.** The strategy comparison
(PRIMER Part 5) so far trained the audio-only model. F6 puts the FusionModel
in the training slot. One design decision is already made: in A/B the edge
ships the **vision feature vector** (256 numbers) instead of the raw frame —
smaller, and not reversible into an image, so frames still never leave the
device. Strategy C is unchanged (all data stays local). Expected shape: same
trade-off as before (bandwidth ↓, edge compute ↑ toward C), with a bigger
model.

**F7 — Pi feasibility.** Time on the real Pi 5: MobileNetV3 feature
extraction per frame, and one epoch of fusion training on a small buffer.
Produces a timings table answering "can the fusion-era edge run on the
hardware?"

Outputs to know: `models/fusion_*.keras`,
`data/vggsound/fusion_training_metrics.json`,
`data/vggsound/fusion_blackout_results.json`.

---

# Appendix — how we found the naive-fusion failure (optional reading)

Kept short; the debugging method is reusable.

Naive fusion (no recipe) scored *below* vision-only, and three cheap probes
showed why: (1) different training runs landed on *suspiciously identical*
accuracies — an "identical number" is a bug smell; (2) with black frames +
real audio the model predicted one constant class for all 98 clips, and
swapping the audio changed 0/20 predictions — audio influence: zero; (3) a
layer probe showed the audio branch's output barely varied between different
sounds — the branch itself was flat.

The concept behind it: **modality gradient starvation.** The pre-trained eye
explains the labels from minute one, the loss drops, and the learning signal
reaching the from-scratch ear shrinks to nothing — it never develops, and the
head wires it out. Modality dropout *alone* couldn't fix this (it can't
revive a branch that produces no signal); the warm-start could. Hence the
order in the recipe: make the ear competent first, then force the head to
rely on it.

Lesson worth keeping: a multimodal model can silently ignore a modality while
looking fine on clean accuracy — you only see it by testing the failure
condition (blackout) and probing inside the model.

---

# Glossary additions (beyond PRIMER's)

- **Feature vector** — a compact numeric summary a branch produces from its
  input (256 numbers for a frame, 128 for a sound); not a label.
- **Late fusion** — encode each sense separately, merge the feature vectors,
  decide on the merged result.
- **Fusion head** — the small trained layers after the merge that map 384
  numbers → class probabilities.
- **Paired data** — (image, audio, label) from the same moment.
- **Leakage** — test data overlapping training data (e.g. windows of one clip
  in both splits) → inflated accuracy. Prevented by splitting by clip.
- **Warm-start** — initialize part of a model from an already-trained model
  instead of random.
- **Modality dropout** — randomly blank one sense during training to force
  competence in the other.
- **Blackout evaluation** — test with one sense disabled to measure graceful
  degradation (our "fog test").
- **Link rot** — YouTube videos in the VGGSound list that no longer exist.
- **Modality gradient starvation** — a pre-trained branch soaks up the
  learning signal; the from-scratch branch never develops (see Appendix).

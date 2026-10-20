# Model-free forensics

Verity's forensic analyzers sit alongside the ML detectors. Each one is a
classic signal-processing or metadata technique whose output an examiner can
explain without reference to a trained model. They **never produce a
real/fake verdict**: they produce *observations* — a sentence plus the numbers
behind it — with a severity that says how much attention the observation
deserves.

They run early in the pipeline (`verity run` and `POST /api/run`), before any
detector, because they are cheap, deterministic and sometimes decisive (an
mismatched EXIF chain can settle a case before the first GPU wakes up).

- **Fail soft by design.** An analyzer error becomes a `notice` observation
  (`"analyzer X failed: …"`) and the examination proceeds. Evidence tools
  must not lose the whole case because one codec quirk tripped one analyzer.
- **Bounded work.** Every analyzer caps its input (frame counts, decoded
  seconds, probe sizes) so long recordings stay cheap.
- **Deterministic.** Same file, same observations.
- **Stored as evidence.** All results land in
  `workspace/artifacts/<media_id>/forensics/forensics.json` and are shown in
  the web console and the report; per-analyzer PNG/JSON artifacts live next to
  it.

Severity is a reading aid, not a verdict:

| Severity | Meaning |
|---|---|
| `info` | neutral fact worth recording (e.g. "no audio track") |
| `notice` | deviates from what an untouched camera file would show, benign explanations still plausible |
| `warning` | internal inconsistency hard to explain without processing or editing |

Observations are flattened and sorted severity-first (then analyzer, then
statement) so the top of the list is always the part that deserves attention.

## The analyzers

Pipeline order: `metadata → compression → noise → video_temporal → audio`
(`src/verity/forensics/runner.py`). Applicability by media type is declared
per analyzer; videos get image analyzers run on sampled frames.

### 1. `metadata` — container, stream and EXIF consistency

**Why.** Every encoder, editor and camera leaves fingerprints in the
container (`encoder` tags, brand codes), the streams (pixel format, profile,
frame-rate mode, GOP structure) and, for photographs, the EXIF block. A file
that claims to come straight from a camera but carries an ffmpeg encoder tag,
a 4:4:4 pixel format or an EXIF modify date earlier than its capture date is
internally inconsistent. **Metadata is trivially stripped or forged** — its
absence proves nothing and its presence is only as trustworthy as the chain
of custody; what the analyzer surfaces is whether the pieces agree.

What it checks:

- *transcoder signatures* — a 60+ entry signature list (ffmpeg/libav, x264/x265,
  Handbrake, Adobe, Final Cut, Canva, CapCut, OBS, Stable Diffusion,
  Midjourney, social platforms …) is matched against `encoder`/`software`
  tags;
- *pixel format and profile* — consumer capture records `yuv420p`-family;
  `4:4:4` or unusual profiles are a notice;
- *GOP structure* — picture types (I/P/B) of the first frames via `ffprobe`;
  all-I (all-intra) or I-frames exactly aligned to keyframe positions is
  characteristic of editing, screen-recordings or freshly encoded content;
  compared with the temporal analyzer's scene cuts;
- *EXIF on photographs* — camera-mandated fields (Make/Model/
  DateTimeOriginal), capture vs. modify dates, Software tag, maintenance of
  orientation data; a *camera-claiming* image missing Make/Model or whose
  modify date precedes the capture date is a warning;
- *aspect ratio* — recorded only when odd;
- *container vs. codec pairing* — brand codes that don't match the box
  contents.

Artifacts: `metadata/metadata.json` (the raw ffprobe dump), `metadata/exif.json`.

Thumb rule for examiners: **a warning here means "pieces disagree", not
"edited"** — transcoding for delivery also rewrites most of these fields.

### 2. `compression` — ELA, quantization tables, double-JPEG, block grid

**Why.** Lossy compression leaves a structured footprint. Every JPEG save
quantizes 8×8 DCT blocks with a known table, so:

- **Error Level Analysis (ELA)** re-saves the image at a fixed quality (90)
  and maps what changed. Content already at that quality changes little;
  freshly pasted or re-rendered regions change more. Flat and highly textured
  regions also respond differently — ELA is *a map to read, not a score to
  trust*.
- **Quantization tables** reveal the encoder's quality setting and whether
  the tables are the standard IJG ones (libjpeg, most phones) or custom
  (Photoshop, some cameras); the closest IJG quality and its L1 distance are
  reported.
- **Double-JPEG periodicity** — compressing once yields a smooth histogram of
  quantized DCT coefficients; a second compression with a different table
  stamps periodic gaps and peaks (Popescu & Farid 2004; Lukás & Fridrich
  2003). The analyzer measures the peak-to-background ratio of the Fourier
  transform of the de-trended histogram over low-frequency AC coefficients.
- **Blockiness** compares pixel differences across 8×8 block boundaries with
  differences elsewhere, per 64 px cell: a pasted JPEG region whose grid
  doesn't align with the host's is a classic splicing trace.

For videos, the first 5 decoded frames are analyzed the same way and reported
as indicative only.

Limits: re-saving the whole image also double-compresses it; heavy social
media re-compression blurs every trace; coarse quantization leaves too few
bins for the periodicity test (reported as "not measurable" rather than 0).

Artifacts: `compression/ela.png`, `compression/blockiness.png`,
`compression/dct_histogram.png` (per-examined-channel), plus the numbers in
`forensics.json` (`periodicity`, `est. quality`, table provenance…).

### 3. `noise` — sensor-noise consistency and spectral peaks

**Why.** A camera sensor adds noise whose level is roughly uniform across a
frame (modulated by brightness and ISO). Content pasted from another picture,
smoothed by a generator or re-rendered by a face-swap model carries a
different noise level, so the *local* estimate becomes inconsistent.

Method:

- residual = image − 3×3 median filter; noise level per 48 px cell is the RMS
  over **non-edge pixels** (Sobel-magnitude threshold keeps edges from being
  mistaken for noise); cells with a robust z-score beyond 3.5 of the picture's
  median level are outlined;
- the **log-magnitude Fourier spectrum** (windowed, centered, up to 1024 px)
  is scanned for isolated peaks; natural images decay smoothly, upsampling
  and generative decoders leave regular grids of peaks (Zhang, Karaman & Chang
  2019; Durall et al. 2020). Codec grids also add peaks at multiples of 1/8
  cycles per pixel — a peak alone is not a verdict.

Limits: textured regions (hair, foliage) have a higher residual than flat
ones regardless of origin; strong phone denoising flattens noise everywhere;
recompression masks both signals.

Artifacts: `noise/noise_map.png`, `noise/spectrum.png`.

### 4. `video_temporal` — cuts, frozen runs, flicker

**Why.** Edits that operate on individual frames leave traces in how
consecutive frames relate to each other. The stream is decoded once at
reduced width (320 px, bounded at 1500 frames) and per frame the **mean
absolute difference to the previous frame**, a 64-bit **perceptual hash**
(low-frequency DCT) and the mean **luminance** are recorded.

From these series:

- *scene-cut candidates* — outlying difference energy (robust z ≥ 6.0 and
  absolute energy ≥ 12). Cuts are normal in edited footage and absent in a
  single take; positions are compared against keyframes from the metadata
  analyzer;
- *frozen / duplicated runs* — consecutive identical hashes with near-zero
  difference energy: frame-rate conversion, dropped frames, or a still frame
  held to cover an edit;
- *flicker* — mean second difference of the luminance series is an
  alternation that face-swap and frame-wise generators produce and cameras do
  not;
- **face-size jitter** — derived from the face extractor's saved crops: only
  the size series is available (boxes aren't stored), so positional jitter is
  *not* computed — recorded so examiners don't assume it.

Limits: a hand-held camera or panning shot raises difference energy
everywhere, so thresholds are relative to the clip itself; one cut in a
relevant place is *an edited video*, which says nothing about the content.

Artifacts: `video_temporal/temporal.json` (per-frame series and thresholds),
`video_temporal/timeline.png` (difference energy and per-frame score with cut
markers).

### 5. `audio` — spectrum, noise floor, spectrogram

**Why.** Synthetic speech (TTS, voice conversion, vocoders) is usually
generated at 16–24 kHz and band-limited: the long-term spectrum has almost no
energy **above 8 kHz** where a microphone recording in a room still carries
noise and sibilance. Generated audio also tends to have an unnaturally low
noise floor between phrases (very clean silence), no DC offset and no
clipping, whereas field recordings show the opposite. The spectrogram lets an
examiner see phrase boundaries, missing breaths, repeated identical segments
or a hard spectral cut-off.

Method: the analyzer decodes the **original** stream at its native sample rate
(the pipeline's audio extractor resamples to 16 kHz, which would hide the very
band this test looks at), up to 180 s, mono PCM. It reports

- long-term frequency spectrum with a **high-band (8 kHz+ vs low-band)
  energy ratio** — below −45 dB the track is *band-limited*;
- RMS level series → silence segments and the **noise floor between phrases**;
- DC offset and headroom (clipping) statistics;
- a 64-band mel spectrogram for the examiner's eyes.

Limits: telephone, Bluetooth and heavily compressed audio are band-limited
for innocent reasons; a clean studio recording also has a low noise floor.
Nothing here identifies a speaker or a generator.

Artifacts: `audio/spectrogram.png`, `audio/spectrum.png`, `audio/stats.json`.

## Where results surface

| Surface | What appears |
|---|---|
| `forensics.json` | full structured record (analyzer → observations + evidence numbers + artifact paths) |
| Web console · Case detail | "Forensic observations" card, severity-sorted, with evidence figures inline |
| HTTP API | `GET /api/media/{id}` returns `forensics` with the same payload; `GET /api/media/{id}/files?path=…` serves the PNG artifacts |
| Report (HTML/PDF) | the observations in severity order, statements only — the examiner reads numbers in the UI |

## Honest boundaries

These analyses are attribution aid, not attribution proof:

- They detect **inconsistency**, not intent. Innocent processing (messaging
  apps, editors, screenshots, rotation) produces several of these signals.
- Their absence is also weak evidence: genAI pipelines that encode at
  24 kHz, phone denoise and re-encode at low quality can wipe every trace.
- Every analyzer documents its sensitivity sources in the `method` string,
  which the report prints verbatim, so a reviewer can challenge the
  reasoning rather than trust a badge.

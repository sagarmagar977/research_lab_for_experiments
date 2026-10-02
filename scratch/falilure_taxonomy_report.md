# Level 2.2 Failure Taxonomy and Gap Analysis

## 1. Frozen Baseline Summary
- **Current Algorithm**: Submodular greedy selection using multiplicative factors (Quality, ViT Diversity, Event Evidence) applied to a base information gain (OCR + Spatial).
- **Stopping Criteria**: Halts when target coverage ($\tau = 0.95$) is reached (`coverage_target_reached`) or marginal information gain drops below a floor (`raw_information_floor`, $< 0.02$).
- **Benchmark**: Method E1 (combining all features) is the current high-water mark at 37.0% F1, evaluated across 112 human keyframes in 4 modalities (Math, Slides, Linear Algebra, Coding). No code changes were made to A–E2 for this diagnosis.

## 2. Failure Taxonomy (Evidence-Based Categories)
Based on the harvest of 43 representative FP/FN cases across 4 sessions, the primary failure modes map to:

- **C. Transient/noisy change being treated as information (Primary cause of False Positives)**
  - *Evidence*: `frame_0030.jpg` (MATH), `frame_0091.png` (Slides), `frame_0454.jpg` (Linear Algebra), `frame_0035.jpg` (Coding).
  - *Mechanism*: Spurious frames achieve a raw gain $\ge 0.02$ due to in-flight typing, pen strokes, cursors, or slide transition artifacts. The algorithm sees these as "new information."

- **D. Another frame satisfying the coverage objective first (Primary cause of False Negatives)**
  - *Evidence*: `frame_0028.jpg` (MATH), `frame_0321.png` (Slides), `frame_0095.jpg` (Linear Algebra).
  - *Mechanism*: The greedy selector accumulates bounding boxes and OCR tokens from earlier (often spurious or intermediate) frames. By the time the true STATE or FINAL keyframe arrives, the coverage is calculated as 95% - 100%, causing the frame to be omitted (marginal gain = 0.0).

- **F. Stopping criterion issue**
  - *Evidence*: The `coverage_target_reached` triggers prematurely for almost all False Negatives in Math, Slides, and Linear Algebra. In Coding (`frame_0078.jpg`), `raw_information_floor` is triggered because coverage is ~84%, but marginal gain is exactly 0.0000.

- **A. Representation gap**
  - *Evidence*: Coding session `frame_0078.jpg` shows a spatial gain of "Subsumed" and OCR gain of 0.0. In coding, new typing happens within the same terminal/editor bounding box. Representing "novelty" purely as non-overlapping spatial boxes or strict OCR token sets fails to capture state changes like a new line of code replacing an old one or a terminal command execution.

- **G. Feature interaction**
  - *Evidence*: In `frame_0078.jpg` (Coding) and `frame_0095.jpg` (Linear Algebra), the `quality_score` (0.833) and `event_evidence` (0.08–0.40) indicate a good frame, but because the base Raw Gain is 0.0, the multiplicative factors cannot rescue the frame. Multiplication by zero yields zero score, effectively masking these strong signals.

## 3. Feature-Space Gap
- **Spatial Subsumption**: The spatial feature treats any bounding box that overlaps with a previous one as "Subsumed." For coding and writing, where action happens in a localized area (e.g. typing inside a terminal window), this zeros out the gain of true state changes.
- **OCR Token Volatility**: OCR reads incomplete words or cursor artifacts as unique tokens. When the final word is completed, the tokens change, but the algorithm might have already spent its "budget" or hit the target coverage on the noisy intermediate steps.

## 4. Objective/Algorithm Gap
- **Greedy Accumulation vs. Global Best**: The algorithm greedily consumes information as it appears chronologically. It lacks the ability to "swap out" a noisy intermediate frame for a cleaner, higher-quality final state frame that contains the same information.
- **Multiplicative Masking**: Relying on $Score = Gain \times Quality \times \dots$ means that if $Gain \to 0$ (due to premature coverage), all other carefully engineered signals (Quality, ViT, Event) are completely ignored.

## 5. Evidence Matrix

| Session | Frame | Type | Actual Human Label | Algorithm Status | Root Cause Category | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MATH** | `frame_0030.jpg` | FP | Spurious | Selected (All Modes) | **C** (Transient Noise) | Gain=0.05-0.09 $\ge$ 0.02. Selected via raw gain. |
| **MATH** | `frame_0028.jpg` | FN | STATE Keyframe | Omitted (All Modes) | **D / F** (Coverage Target) | Gain=0.0. Target coverage reached 99%-100% early. |
| **Slides** | `frame_0091.png` | FP | Spurious | Selected (D, E1, E2) | **C** (Transient Noise) | Gain=0.04 $\ge$ 0.02. Selected via raw gain. |
| **Slides** | `frame_0321.png` | FN | FINAL Keyframe | Omitted (All Modes) | **D / F** (Coverage Target) | Gain=0.0. Target coverage 95.31% reached early. |
| **Lin_Alg**| `frame_0454.jpg` | FP | Spurious | Selected (D, E1, E2) | **C** (Transient Noise) | Gain=0.03 $\ge$ 0.02. |
| **Lin_Alg**| `frame_0095.jpg` | FN | STATE Keyframe | Omitted (All Modes) | **D / F** (Coverage Target) | Gain=0.0. Subsumed. Target coverage 97.57% reached. |
| **Coding** | `frame_0035.jpg` | FP | Spurious | Selected (All Modes) | **C** (Transient Noise) | Gain=0.02 $\ge$ 0.02. |
| **Coding** | `frame_0078.jpg` | FN | STATE Keyframe | Omitted (All Modes) | **A / B / G** (Representation) | Gain=0.0. Spatial=Subsumed. Quality=0.833 ignored. |

## 6. Research Implication
1. **Redefining "Gain"**: We cannot strictly use monotonic accumulation of bounding boxes/tokens as the sole definition of "information." We need a way to measure *state stability* rather than just *incremental area*.
2. **Post-Processing / Pruning**: A purely chronological greedy algorithm cannot easily distinguish an intermediate noisy frame from a final state until the final state arrives. We likely need a two-pass approach or a substitution mechanism (e.g., if frame B subsumes frame A's information but has higher quality, replace A with B).
3. **Additive vs Multiplicative**: If a frame is a known "Event" or has very high "Quality", it should perhaps have a chance to be selected even if the purely lexical "Raw Gain" is near zero due to coverage exhaustion.

## 7. What We Should NOT Change Yet
- **Do not discard the existing features**: ViT, SSIM, Quality, and Event Evidence are actually producing good numbers (e.g., Quality = 0.833 for Coding FN, 0.97 for Slides FN). The features are not the problem; how they are combined (the greedy objective) is.
- **Do not train a new model**: The signals required to make the right choice are already present in the data. The failure is structural (algorithmic logic), not a lack of latent feature representation.
- **Do not change the 0.02 floor or 0.95 target arbitrarily**: The issue isn't that the thresholds are slightly mis-tuned; the issue is that noisy frames hit the threshold too easily, and good frames are blocked by the rigid greedy coverage logic.

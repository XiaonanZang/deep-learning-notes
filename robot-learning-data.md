---
title: Robot Learning Data
---

# Robot Learning Data — Datasets, Demonstrations, and Policy-Fit

The data side of imitation learning for manipulation: how demonstrations are collected, the
key public datasets, and what makes a dataset actually improve a policy.

---

## 1. Imitation learning in one paragraph

The dominant recipe for robot manipulation today is **behavior cloning**: collect
demonstrations of a task (observations paired with the expert's actions), then train a policy
to predict the action given the observation. It is supervised learning where the label is "what
the expert did next." Modern policies (diffusion policies, vision-language-action models) are
fancier about how they represent the action distribution, but the data contract is the same:
`(observation → action)` pairs, in sequence, from a trusted demonstrator.

Because it is supervised, everything about the *data* propagates straight into the policy. That
is why the data, not the loss function, is where the effort goes.

## 2. How demonstrations are collected

- **Teleoperation.** A human drives the robot (VR controllers, a leader-follower arm, a
  space-mouse). You get perfectly on-embodiment data (real robot states and actions) but at
  human speed, so it is expensive and slow to scale.
- **Kinesthetic teaching.** Physically moving the robot through the motion. Simple, low
  throughput.
- **Egocentric human video.** Film a person doing the task from a first-person view. Abundant
  and cheap, but the actions are human-hand, not robot-gripper, so there is an **embodiment gap**
  to bridge.
- **Simulation.** Generate demos in a simulator. Scales infinitely but carries a **sim-to-real
  gap**.
- **Autonomous / fleet.** Deployed robots log their own runs. Scales with deployment but the
  data is only as good as the current policy.

## 3. The key datasets (the vocabulary to know)

These are the names that come up when people talk about manipulation data. Know what each *is*
and what it is *for*.

| Dataset | What it is | Why it matters |
|---|---|---|
| **DROID** | Large-scale, in-the-wild teleoperated manipulation across many labs, scenes, and tasks | The reference for *diverse, real, on-robot* demonstration data. Diversity over a single lab. |
| **Open X-Embodiment / RT-X** | A pooled corpus across many robot embodiments | Showed that training across embodiments transfers; the case for cross-embodiment data. |
| **Ego4D** | Massive first-person human video (thousands of hours, everyday activity) | The canonical egocentric *human* video corpus; the source side of learning-from-human-video. |
| **EgoMimic** | Framework for learning manipulation from egocentric human video, bridging to a robot | A concrete instance of closing the human-to-robot embodiment gap. |
| **MimicLabs / data-composition studies** | Systematic study of *which* data mixtures improve policies | Evidence that composition and diversity, not raw count, drive policy performance. |

The through-line: the frontier moved from "collect more of one lab's data" to "compose diverse,
cross-embodiment, human-plus-robot data and measure what actually helps."

## 4. What makes data "policy-fit"

"Policy-fit" is the useful piece of jargon: data that measurably improves the downstream policy,
as opposed to data that is merely large or well-formatted. The dimensions people validate on:

1. **Schema conformance.** Every episode has synchronized, correctly typed streams: video,
   robot state, actions, timestamps, calibration. Break this and training silently corrupts.
2. **Annotation quality.** Masks, tracks, language instructions, success/fail flags are correct
   and consistent. Label noise caps the ceiling.
3. **Embodiment coverage.** The action space and morphology are represented and aligned. A
   two-finger gripper policy does not learn from five-finger-hand data without alignment.
4. **Distributional diversity.** Objects, poses, lighting, backgrounds, and especially
   **recovery/correction** states. This is what generalization is made of.
5. **Provenance and versioning.** You can trace every sample to its source, checkpoint, and
   preprocessing, so results are reproducible and bad batches are removable.

A dataset that scores well on these five moves a policy more than a bigger dataset that does not.
Measuring and selling that difference is the entire premise of a data-curation layer.

## 5. The embodiment gap (the central technical problem)

Human video is cheap; robot data is expensive. So the field wants to learn from human video and
transfer to robots. The obstacle is that a human hand and a robot gripper have different
morphologies and action spaces. Bridging strategies:

- **Retarget** the human motion onto the robot's kinematics.
- **Learn a shared latent** action space so both map into the same representation.
- **Use human video for perception/affordances** (what to touch, where) and robot data for the
  low-level control.

You do not have to solve this to work in the data layer, but you have to *speak* it: it is why
egocentric video, cross-embodiment alignment, and correction data keep coming up.

## 6. Where this connects

- The upstream picture: **[The Physical AI Data Layer](physical-ai-data-layer.html)**.
- How raw video becomes searchable, labeled, quality-scored evidence:
  **[Auto-labeling & the Evidence Pipeline](auto-labeling-evidence-pipeline.html)**.
- The learning machinery behind the policies:
  **[Transformers](transformers.html)**, **[Vision Transformers](vision-transformers.html)**,
  and the **[decoder / autoregressive loop](transformer-decoder.html)** (VLAs generate action
  tokens the same way a language model generates text tokens).

---

*The one-line summary: imitation learning copies demonstrations, so a policy inherits its data's
quality, diversity, and embodiment coverage. "Policy-fit" data (clean, diverse, recovery-rich,
provenance-tracked) beats bigger dirty data, and closing the human-video-to-robot embodiment gap
is the field's central lever.*

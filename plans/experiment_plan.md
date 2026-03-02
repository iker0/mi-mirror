## Goal 
Use Mechanistic Interpretability techniques to investigate how image generation models create reflection in mirrors in the generated images.

## Hypothesis
Similar to the Induction Head in LLM, we can find a Reflection Head or Reflection Circuit. 

## Experiments
Locate the Reflection circuit and Reflection head
### Step 1 -- Selectivity Score
For each head, compute a Reflection Selectivity Score measuring how much more it attends cross-region in mirror images vs. non-mirror images:
- For mirror images, compute `CRA_mirror` = average cross-region attention (reflection ROI -> object ROI)
- For non-mirror images, compute `CRA_nonmirror` = average cross-region attention
- Selectivity Score = `S(h, l) = CRA_mirror(h, l) - CRA_nonmirror(h, l)`
- Rank all heads by Selectivity Score. Heads with high `S` are candidate reflection heads 

## Step 2 -- Attention entropy analysis
- `H(h, l, t) = - sum_ij A * log A`
- Reflection heads should show lower entropy on mirror images.
- Combine with selectivity: HIES score = `S x (H_nonmirror - H_mirror) / H_max`

### Step 3 -- Temporal profiling
- Plot `CRA(h, l, t)` as a function of denoising timestep `t` 
- Identify the critical window: timestep range where cross-region attention peaks.

### Step 4 -- Spatial pattern analysis
Visualize what candidate heads actually attend to:
- For each candidate head, render the attention map as a heatmap: for query tokens in reflection region `R_ref`, show which key tokens they attend to.

### Step 5 -- Causal validation
- For each candidate reflection head, zero-ablate it: set its output contribution to zero during inference on mirror prompts. Measure degradation in reflection quality.
- A head is confirmed reflection head if ablating it significantly degrades reflection quality but not overall image quality.

### Step 6 -- Circuit composition analysis
Reflection heads likely don't work in isolation. Investigate multi-head circuits.
- Ablate pairs of heads (one early, one late candidate) and measure if the joint effect matters more than the sum of individual ablations.
- This may reveal a reflection circuit e.g., Head A (layer 4) identifies the object -> Head B (layer 8) identifies the mirror boundary -> Head C (layer 12) copies features from object region to mirror region with spatial transformation.


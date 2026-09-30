# `data/whole_clip.py` — saved native D0 input contract

## Objective

Reconstruct exactly the noised whole-video input used in a saved baseline D0 run. Calibration uses the baseline alone; a candidate is needed only for a paired comparison.

## Data flow

`load_manifest` validates the one-step white-capture, full-bidirectional run. `build_input` checks the capture content hash and fps, loads the saved epsilon, verifies the token shape and saved geometry, and rebuilds the clean-first-frame modality with the recorded sigma and text context. It returns the grid, modality, clean first-frame tokens and source row. `verify_pair` adds candidate-versus-baseline setup checks; `verify_saved_noise` compares the actual epsilon tensors. `latent_path` resolves the saved D0 output.

## Invariants and checks

Frame 0 is conditioning, not a predicted frame. Sigmas have exact `[sigma, 0]` schedules. Do not derive fps from defaults or replace the saved epsilon with a new draw. The module has no dependence on a particular candidate checkpoint. `test_whole_clip_d0.py` covers manifest rejection; real forward equality to the saved baseline checks the model-facing tensor contract after changes.

# Architecture Notes

## Initial design

1. **Ingestion:** accept a local match video and validate codec, dimensions, frame rate, and duration.
2. **Frame processing:** decode frames with OpenCV; optionally sample frames for inference while retaining timestamps.
3. **Detection:** run a pretrained detector for player/person candidates and evaluate a football-specific ball model separately.
4. **Tracking:** pass frame detections into a multi-object tracker; persist track IDs, confidence, bounding boxes, and timestamps.
5. **Analytics:** classify teams from player crops; estimate ball-player association and possession; aggregate track positions.
6. **Visualization:** render annotated video, per-player movement plots, heatmaps, and (after calibration) a 2D pitch view.
7. **Evaluation:** compare detections and tracks with ground-truth annotations on held-out match footage.

## Engineering rules

- Keep detection, tracking, analytics, and rendering in separate modules.
- Preserve timestamps when sampling frames; do not treat sampled frame indices as original video frame numbers.
- Store raw model outputs before deriving statistics so analytics can be recomputed.
- Mark uncertain ball detections and possession estimates rather than silently inventing values.
- Record model weights, configuration, dataset split, and software versions with every experiment.
- Do not claim real-world speed or distance without a documented calibration method.

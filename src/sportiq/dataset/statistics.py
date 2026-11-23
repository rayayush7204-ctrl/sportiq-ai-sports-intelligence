"""Dataset statistics generation, class balance evaluation, and track length analytics."""

from __future__ import annotations

import statistics as py_stats
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sportiq.dataset.annotation_parser import SequenceDataset
from sportiq.dataset.class_mapping import ClassTaxonomyType, resolve_class


@dataclass
class DatasetStatistics:
    """Comprehensive metrics and distributions across inspected sequences."""

    total_sequences: int = 0
    sequence_names: List[str] = field(default_factory=list)
    total_frames: int = 0
    total_annotations: int = 0
    unique_tracks: int = 0
    raw_class_distribution: Dict[str, int] = field(default_factory=dict)
    canonical_class_distribution: Dict[str, int] = field(default_factory=dict)
    staff_annotation_count: int = 0
    staff_percentage: float = 0.0
    track_length_metrics: Dict[str, float] = field(default_factory=dict)
    bbox_dimension_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sequences": self.total_sequences,
            "sequence_names": self.sequence_names,
            "total_frames": self.total_frames,
            "total_annotations": self.total_annotations,
            "unique_tracks": self.unique_tracks,
            "raw_class_distribution": self.raw_class_distribution,
            "canonical_class_distribution": self.canonical_class_distribution,
            "staff_annotation_count": self.staff_annotation_count,
            "staff_percentage": self.staff_percentage,
            "track_length_metrics": self.track_length_metrics,
            "bbox_dimension_metrics": self.bbox_dimension_metrics,
        }


class DatasetStatisticsReporter:
    """Analyzes inspected sequences and compiles ground-truth distributions."""

    def __init__(self, taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS) -> None:
        self.taxonomy = taxonomy

    def analyze(self, sequences: List[SequenceDataset]) -> DatasetStatistics:
        """Compute empirical statistics across a collection of SequenceDataset objects.

        Args:
            sequences: List of parsed sequence datasets.

        Returns:
            DatasetStatistics: Aggregated statistics.
        """
        if not sequences:
            return DatasetStatistics()

        total_sequences = len(sequences)
        sequence_names = [s.name for s in sequences]
        total_frames = sum(s.num_frames for s in sequences)
        total_annotations = sum(s.total_annotations for s in sequences)

        raw_counter: Counter[str] = Counter()
        canonical_counter: Counter[str] = Counter()

        all_widths: List[float] = []
        all_heights: List[float] = []
        all_aspect_ratios: List[float] = []

        # Track lengths: map (seq_name, track_id) to frame count
        track_lengths: Dict[tuple, int] = Counter()

        for seq in sequences:
            for ann in seq.annotations:
                # Track length
                track_lengths[(seq.name, ann.track_id)] += 1

                # Dimensions
                if ann.bb_width > 0 and ann.bb_height > 0:
                    all_widths.append(ann.bb_width)
                    all_heights.append(ann.bb_height)
                    all_aspect_ratios.append(ann.bb_width / ann.bb_height)

                # Class resolution
                raw_class = "unclassified"
                if seq.game_info and ann.track_id in seq.game_info.track_classes:
                    raw_class = seq.game_info.track_classes[ann.track_id]

                _, canonical_name, raw_norm = resolve_class(raw_class, taxonomy=self.taxonomy)
                raw_counter[raw_norm] += 1
                canonical_counter[canonical_name] += 1

        unique_tracks = len(track_lengths)
        lengths_list = list(track_lengths.values()) if track_lengths else [0]

        staff_count = canonical_counter.get("staff", 0)
        staff_pct = round((staff_count / total_annotations * 100), 2) if total_annotations > 0 else 0.0

        track_metrics = {
            "min_frames": float(min(lengths_list)),
            "max_frames": float(max(lengths_list)),
            "mean_frames": round(float(py_stats.mean(lengths_list)), 2),
            "median_frames": round(float(py_stats.median(lengths_list)), 2),
        }

        bbox_metrics = {
            "mean_width_px": round(float(py_stats.mean(all_widths)), 2) if all_widths else 0.0,
            "mean_height_px": round(float(py_stats.mean(all_heights)), 2) if all_heights else 0.0,
            "mean_aspect_ratio": round(float(py_stats.mean(all_aspect_ratios)), 3) if all_aspect_ratios else 0.0,
            "min_width_px": round(float(min(all_widths)), 2) if all_widths else 0.0,
            "max_width_px": round(float(max(all_widths)), 2) if all_widths else 0.0,
            "min_height_px": round(float(min(all_heights)), 2) if all_heights else 0.0,
            "max_height_px": round(float(max(all_heights)), 2) if all_heights else 0.0,
        }

        return DatasetStatistics(
            total_sequences=total_sequences,
            sequence_names=sequence_names,
            total_frames=total_frames,
            total_annotations=total_annotations,
            unique_tracks=unique_tracks,
            raw_class_distribution=dict(raw_counter.most_common()),
            canonical_class_distribution=dict(canonical_counter.most_common()),
            staff_annotation_count=staff_count,
            staff_percentage=staff_pct,
            track_length_metrics=track_metrics,
            bbox_dimension_metrics=bbox_metrics,
        )

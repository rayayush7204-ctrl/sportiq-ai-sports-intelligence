"""Unit tests for dataset ground-truth statistics compilation and metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from sportiq.dataset.annotation_parser import load_sequence
from sportiq.dataset.class_mapping import ClassTaxonomyType
from sportiq.dataset.statistics import DatasetStatistics, DatasetStatisticsReporter


def test_statistics_single_sequence(synthetic_mot_sequence: Path) -> None:
    """Test statistics compiled on synthetic sequence."""
    seq = load_sequence(synthetic_mot_sequence)
    reporter = DatasetStatisticsReporter(taxonomy=ClassTaxonomyType.FOUR_CLASS)

    stats = reporter.analyze([seq])

    assert isinstance(stats, DatasetStatistics)
    assert stats.total_sequences == 1
    assert stats.sequence_names == ["SNMOT-SYNTH-01"]
    assert stats.total_frames == 2
    assert stats.total_annotations == 10
    assert stats.unique_tracks == 5

    # Canonical class distribution
    assert stats.canonical_class_distribution["player"] == 2
    assert stats.canonical_class_distribution["goalkeeper"] == 2
    assert stats.canonical_class_distribution["referee"] == 2
    assert stats.canonical_class_distribution["ball"] == 2
    assert stats.canonical_class_distribution["staff"] == 2

    # Staff analytics
    assert stats.staff_annotation_count == 2
    assert stats.staff_percentage == 20.0

    # Track length metrics (each track has 2 frames)
    assert stats.track_length_metrics["min_frames"] == 2.0
    assert stats.track_length_metrics["max_frames"] == 2.0
    assert stats.track_length_metrics["mean_frames"] == 2.0
    assert stats.track_length_metrics["median_frames"] == 2.0

    # Bbox dimensions
    assert stats.bbox_dimension_metrics["min_width_px"] == 15.0  # Ball
    assert stats.bbox_dimension_metrics["max_width_px"] == 60.0  # Goalkeeper


def test_statistics_empty_list() -> None:
    """Test statistics on empty sequence list returns empty statistics structure."""
    reporter = DatasetStatisticsReporter()
    stats = reporter.analyze([])
    assert stats.total_sequences == 0
    assert stats.total_annotations == 0

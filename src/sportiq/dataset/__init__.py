"""SportIQ Dataset Engineering Package.

Provides tools for SoccerNet MOT annotation parsing, validation, class mapping,
YOLO format conversion, deterministic match-level dataset splitting, and statistics.
"""

from sportiq.dataset.annotation_parser import (
    GameInfo,
    MOTAnnotation,
    SequenceDataset,
    SequenceInfo,
    load_sequence,
    parse_gameinfo,
    parse_mot_gt,
    parse_seqinfo,
)
from sportiq.dataset.annotation_validator import AnnotationValidator, ValidationResult
from sportiq.dataset.class_mapping import (
    ClassTaxonomy,
    ClassTaxonomyType,
    resolve_class,
)
from sportiq.dataset.dataset_splitter import DatasetSplit, DatasetSplitter
from sportiq.dataset.downloader import SoccerNetDownloaderHelper
from sportiq.dataset.statistics import DatasetStatistics, DatasetStatisticsReporter
from sportiq.dataset.visualizer import CLASS_COLORS_BGR, draw_annotations, visualize_sequence_frame
from sportiq.dataset.yolo_converter import YOLOConverter, YOLOInstance

__all__ = [
    "MOTAnnotation",
    "SequenceInfo",
    "GameInfo",
    "SequenceDataset",
    "parse_mot_gt",
    "parse_seqinfo",
    "parse_gameinfo",
    "load_sequence",
    "AnnotationValidator",
    "ValidationResult",
    "ClassTaxonomy",
    "ClassTaxonomyType",
    "resolve_class",
    "YOLOConverter",
    "YOLOInstance",
    "DatasetSplitter",
    "DatasetSplit",
    "DatasetStatisticsReporter",
    "DatasetStatistics",
    "SoccerNetDownloaderHelper",
    "draw_annotations",
    "visualize_sequence_frame",
    "CLASS_COLORS_BGR",
]

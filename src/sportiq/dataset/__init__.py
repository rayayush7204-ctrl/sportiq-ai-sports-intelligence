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
from sportiq.dataset.catalog import (
    MatchCatalogGroup,
    SequenceCatalogEntry,
    SoccerNetCatalog,
    build_soccernet_catalog,
    group_sequences_by_game,
)
from sportiq.dataset.class_mapping import (
    ClassTaxonomy,
    ClassTaxonomyType,
    resolve_class,
)
from sportiq.dataset.dataset_materializer import (
    AlignmentReport,
    AlignmentVerificationError,
    DatasetMaterializer,
)
from sportiq.dataset.dataset_splitter import DatasetSplit, DatasetSplitter
from sportiq.dataset.downloader import SoccerNetDownloaderHelper
from sportiq.dataset.frame_acquisition import (
    INTERNAL_SPLIT_MATCHES,
    DatasetIntegrityReport,
    ImageIntegrityAuditor,
    MatchDatasetMaterializer,
    SequenceIntegrityReport,
    StorageBudgetChecker,
    ZipPathSafetyValidator,
)
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
    "DatasetMaterializer",
    "AlignmentReport",
    "AlignmentVerificationError",
    "DatasetStatisticsReporter",
    "DatasetStatistics",
    "SoccerNetDownloaderHelper",
    "draw_annotations",
    "visualize_sequence_frame",
    "CLASS_COLORS_BGR",
    "SequenceCatalogEntry",
    "MatchCatalogGroup",
    "SoccerNetCatalog",
    "build_soccernet_catalog",
    "group_sequences_by_game",
    "ZipPathSafetyValidator",
    "StorageBudgetChecker",
    "ImageIntegrityAuditor",
    "MatchDatasetMaterializer",
    "SequenceIntegrityReport",
    "DatasetIntegrityReport",
    "INTERNAL_SPLIT_MATCHES",
]

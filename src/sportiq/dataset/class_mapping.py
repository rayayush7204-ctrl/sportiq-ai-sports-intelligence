"""Taxonomy definitions and semantic class mapping for football object detection.

Taxonomies:
  - FOUR_CLASS (Primary):
      0: player
      1: goalkeeper
      2: referee
      3: ball
  - FIVE_CLASS (Experimental, with staff):
      0: player
      1: goalkeeper
      2: referee
      3: ball
      4: staff
  - TWO_CLASS (Person/Ball Baseline):
      0: person
      1: ball
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple


class ClassTaxonomyType(str, Enum):
    """Supported detection taxonomy models."""

    FOUR_CLASS = "four_class"
    FIVE_CLASS = "five_class"
    TWO_CLASS = "two_class"


class ClassTaxonomy:
    """Class label mappings and schema definitions."""

    FOUR_CLASS_NAMES: Dict[int, str] = {
        0: "player",
        1: "goalkeeper",
        2: "referee",
        3: "ball",
    }

    FIVE_CLASS_NAMES: Dict[int, str] = {
        0: "player",
        1: "goalkeeper",
        2: "referee",
        3: "ball",
        4: "staff",
    }

    TWO_CLASS_NAMES: Dict[int, str] = {
        0: "person",
        1: "ball",
    }

    @classmethod
    def get_names(cls, taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS) -> Dict[int, str]:
        """Return class ID to name dictionary for specified taxonomy."""
        if taxonomy == ClassTaxonomyType.FOUR_CLASS:
            return cls.FOUR_CLASS_NAMES
        elif taxonomy == ClassTaxonomyType.FIVE_CLASS:
            return cls.FIVE_CLASS_NAMES
        elif taxonomy == ClassTaxonomyType.TWO_CLASS:
            return cls.TWO_CLASS_NAMES
        raise ValueError(f"Unsupported taxonomy: {taxonomy}")


def normalize_class_name(raw_class: Optional[str]) -> str:
    """Normalize raw class string from gameinfo.ini or annotation source."""
    if not raw_class:
        return "unclassified"
    cleaned = raw_class.strip().lower()
    # Normalize punctuation/underscores to spaces
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def resolve_class(
    raw_class: Optional[str],
    taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS,
) -> Tuple[Optional[int], str, str]:
    """Map raw SoccerNet annotation class to detection category ID and canonical name.

    Mapping rules:
      - 'player team left' / 'player team right' / 'player' -> 'player'
      - 'goalkeeper team left' / 'goalkeeper team right' / 'goalkeeper' -> 'goalkeeper'
      - 'main referee' / 'side referee' / 'referee' -> 'referee'
      - 'ball' -> 'ball'
      - 'staff' -> 'staff' (preserved in metadata; unmapped in 4-class, class 4 in 5-class, person in 2-class)

    Returns:
        Tuple[Optional[int], str, str]:
          - class_id: Integer class ID for YOLO training (or None if unmapped / ignored)
          - canonical_name: Standardized category name
          - raw_normalized: Normalized raw class string preserved for lineage
    """
    raw_norm = normalize_class_name(raw_class)

    # 1. Identify canonical semantic entity
    if "goalkeeper" in raw_norm:
        canonical = "goalkeeper"
    elif "player" in raw_norm:
        canonical = "player"
    elif "referee" in raw_norm:
        canonical = "referee"
    elif "ball" in raw_norm:
        canonical = "ball"
    elif "staff" in raw_norm:
        canonical = "staff"
    else:
        canonical = "unclassified"

    # 2. Map canonical entity to the selected taxonomy
    if taxonomy == ClassTaxonomyType.FOUR_CLASS:
        # In four-class detection, staff is kept in metadata but unmapped (None) for YOLO detector
        mapping = {
            "player": 0,
            "goalkeeper": 1,
            "referee": 2,
            "ball": 3,
            "staff": None,
            "unclassified": None,
        }
        return mapping[canonical], canonical, raw_norm

    elif taxonomy == ClassTaxonomyType.FIVE_CLASS:
        mapping = {
            "player": 0,
            "goalkeeper": 1,
            "referee": 2,
            "ball": 3,
            "staff": 4,
            "unclassified": None,
        }
        return mapping[canonical], canonical, raw_norm

    elif taxonomy == ClassTaxonomyType.TWO_CLASS:
        # Person vs Ball baseline
        if canonical in {"player", "goalkeeper", "referee", "staff"}:
            return 0, "person", raw_norm
        elif canonical == "ball":
            return 1, "ball", raw_norm
        else:
            return None, "unclassified", raw_norm

    raise ValueError(f"Unknown taxonomy: {taxonomy}")

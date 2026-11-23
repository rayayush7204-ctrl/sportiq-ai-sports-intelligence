"""Unit tests for football class taxonomy resolution and mapping."""

from __future__ import annotations

import pytest

from sportiq.dataset.class_mapping import (
    ClassTaxonomy,
    ClassTaxonomyType,
    normalize_class_name,
    resolve_class,
)


def test_normalize_class_name() -> None:
    """Test normalization of strings with whitespace and punctuation."""
    assert normalize_class_name("  Player_Team-Left  ") == "player team left"
    assert normalize_class_name(None) == "unclassified"
    assert normalize_class_name("") == "unclassified"


def test_four_class_mapping() -> None:
    """Test Option A 4-class primary taxonomy mapping."""
    # Player
    cid, cname, raw = resolve_class("player team left", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 0
    assert cname == "player"
    assert raw == "player team left"

    cid, cname, _ = resolve_class("player team right", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 0
    assert cname == "player"

    # Goalkeeper
    cid, cname, _ = resolve_class("goalkeeper team left", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 1
    assert cname == "goalkeeper"

    cid, cname, _ = resolve_class("goalkeeper", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 1
    assert cname == "goalkeeper"

    # Referee
    cid, cname, _ = resolve_class("main referee", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 2
    assert cname == "referee"

    cid, cname, _ = resolve_class("side referee", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 2
    assert cname == "referee"

    # Ball
    cid, cname, _ = resolve_class("ball", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid == 3
    assert cname == "ball"

    # Staff is preserved in metadata as canonical 'staff', but class_id is None (unmapped for detector)
    cid, cname, raw = resolve_class("staff", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid is None
    assert cname == "staff"
    assert raw == "staff"

    # Unknown
    cid, cname, raw = resolve_class("crowd_fan", taxonomy=ClassTaxonomyType.FOUR_CLASS)
    assert cid is None
    assert cname == "unclassified"


def test_five_class_mapping() -> None:
    """Test 5-class taxonomy with staff as explicit class 4."""
    cid, cname, raw = resolve_class("staff", taxonomy=ClassTaxonomyType.FIVE_CLASS)
    assert cid == 4
    assert cname == "staff"
    assert raw == "staff"

    cid, cname, _ = resolve_class("player", taxonomy=ClassTaxonomyType.FIVE_CLASS)
    assert cid == 0


def test_two_class_baseline_mapping() -> None:
    """Test 2-class baseline taxonomy: person (0) vs ball (1)."""
    # Player -> person
    cid, cname, _ = resolve_class("player team left", taxonomy=ClassTaxonomyType.TWO_CLASS)
    assert cid == 0
    assert cname == "person"

    # Goalkeeper -> person
    cid, cname, _ = resolve_class("goalkeeper", taxonomy=ClassTaxonomyType.TWO_CLASS)
    assert cid == 0
    assert cname == "person"

    # Referee -> person
    cid, cname, _ = resolve_class("main referee", taxonomy=ClassTaxonomyType.TWO_CLASS)
    assert cid == 0
    assert cname == "person"

    # Staff -> person
    cid, cname, _ = resolve_class("staff", taxonomy=ClassTaxonomyType.TWO_CLASS)
    assert cid == 0
    assert cname == "person"

    # Ball -> ball (1)
    cid, cname, _ = resolve_class("ball", taxonomy=ClassTaxonomyType.TWO_CLASS)
    assert cid == 1
    assert cname == "ball"


def test_taxonomy_names_dictionary() -> None:
    """Test ClassTaxonomy.get_names returns correct dictionaries."""
    four_names = ClassTaxonomy.get_names(ClassTaxonomyType.FOUR_CLASS)
    assert four_names == {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

    five_names = ClassTaxonomy.get_names(ClassTaxonomyType.FIVE_CLASS)
    assert five_names[4] == "staff"

    two_names = ClassTaxonomy.get_names(ClassTaxonomyType.TWO_CLASS)
    assert two_names == {0: "person", 1: "ball"}

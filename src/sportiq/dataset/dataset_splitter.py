"""Sequence-level and match-level dataset splitting to guarantee zero frame leakage."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from sportiq.dataset.annotation_parser import SequenceDataset
from sportiq.dataset.class_mapping import ClassTaxonomy, ClassTaxonomyType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    """Sequence partition mapping."""

    train_sequences: List[str]
    val_sequences: List[str]
    test_sequences: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_sequences": self.train_sequences,
            "val_sequences": self.val_sequences,
            "test_sequences": self.test_sequences,
            "train_count": len(self.train_sequences),
            "val_count": len(self.val_sequences),
            "test_count": len(self.test_sequences),
            "total_count": len(self.train_sequences) + len(self.val_sequences) + len(self.test_sequences),
        }


def compute_split_counts(
    n_items: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    """Compute deterministic item counts for (train, val, test) using largest remainder method."""
    if n_items <= 0:
        return 0, 0, 0

    ratios = [train_ratio, val_ratio, test_ratio]
    exact = [n_items * r for r in ratios]
    counts = [int(x) for x in exact]
    remainders = [(exact[i] - counts[i], i) for i in range(3)]
    remainders.sort(key=lambda x: x[0], reverse=True)

    leftover = n_items - sum(counts)
    for _, idx in remainders:
        if leftover <= 0:
            break
        if ratios[idx] > 0:
            counts[idx] += 1
            leftover -= 1

    # Ensure train has at least 1 item if train_ratio > 0
    if train_ratio > 0 and counts[0] == 0 and n_items >= 1:
        counts[0] = 1
        if counts[1] >= counts[2] and counts[1] > 0:
            counts[1] -= 1
        elif counts[2] > 0:
            counts[2] -= 1

    return counts[0], counts[1], counts[2]


class DatasetSplitter:
    """Partition sequences deterministically with zero data leakage."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize splitter.

        Args:
            random_seed: Random seed for deterministic reproducibility.
        """
        self.random_seed = random_seed

    def split_sequences(
        self,
        sequence_names: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        match_grouping: Optional[Dict[str, str]] = None,
        strict_match_grouping: bool = True,
    ) -> DatasetSplit:
        """Split sequences into train, validation, and test sets.

        Args:
            sequence_names: List of unique sequence names (e.g. ['SNMOT-001', 'SNMOT-002']).
            train_ratio: Proportion of sequences for training.
            val_ratio: Proportion for validation.
            test_ratio: Proportion for testing.
            match_grouping: Optional mapping from sequence_name to match_id/game_id.
                            If provided, all sequences from the same match are kept together.
            strict_match_grouping: If True and match_grouping is provided, raises ValueError
                                   when any sequence lacks a valid game_id.
                                   If False, warns and downgrades to sequence-level splitting.

        Returns:
            DatasetSplit: Deterministic, non-overlapping partition.

        Raises:
            ValueError: If ratios don't sum to 1.0, sequence list is empty, or match metadata is incomplete.
        """
        if not sequence_names:
            raise ValueError("sequence_names list cannot be empty.")

        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 1e-4:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
            )

        # Deduplicate while preserving order
        unique_seqs = sorted(list(set(sequence_names)))
        rng = random.Random(self.random_seed)

        # Validate match_grouping completeness if provided
        active_match_grouping = match_grouping
        if active_match_grouping is not None:
            missing_matches = [
                seq
                for seq in unique_seqs
                if not active_match_grouping.get(seq) or not str(active_match_grouping[seq]).strip()
            ]
            if missing_matches:
                err_msg = (
                    f"Incomplete match metadata: {len(missing_matches)} sequence(s) missing valid game_id: {missing_matches}."
                )
                if strict_match_grouping:
                    raise ValueError(
                        f"{err_msg} When match grouping is enabled, every sequence must have a valid game_id "
                        "to guarantee zero match-level data leakage. Set strict_match_grouping=False to allow fallback."
                    )
                else:
                    logger.warning(
                        "%s Downgrading to sequence-level splitting for all %d sequences.",
                        err_msg,
                        len(unique_seqs),
                    )
                    active_match_grouping = None

        if active_match_grouping:
            # Group sequences by match ID
            groups: Dict[str, List[str]] = {}
            for seq in unique_seqs:
                gid = str(active_match_grouping[seq]).strip()
                groups.setdefault(gid, []).append(seq)

            group_keys = sorted(list(groups.keys()))
            rng.shuffle(group_keys)

            n_train, n_val, _ = compute_split_counts(
                len(group_keys), train_ratio, val_ratio, test_ratio
            )

            train_groups = set(group_keys[:n_train])
            val_groups = set(group_keys[n_train : n_train + n_val])
            test_groups = set(group_keys[n_train + n_val :])

            train_seqs = [s for g in train_groups for s in groups[g]]
            val_seqs = [s for g in val_groups for s in groups[g]]
            test_seqs = [s for g in test_groups for s in groups[g]]
        else:
            # Sequence-level splitting
            seqs_shuffled = unique_seqs.copy()
            rng.shuffle(seqs_shuffled)

            n_train, n_val, _ = compute_split_counts(
                len(seqs_shuffled), train_ratio, val_ratio, test_ratio
            )

            train_seqs = seqs_shuffled[:n_train]
            val_seqs = seqs_shuffled[n_train : n_train + n_val]
            test_seqs = seqs_shuffled[n_train + n_val :]

        # Guarantee zero leakage
        s_train = set(train_seqs)
        s_val = set(val_seqs)
        s_test = set(test_seqs)

        assert not (s_train & s_val), f"Data leakage between train and val: {s_train & s_val}"
        assert not (s_train & s_test), f"Data leakage between train and test: {s_train & s_test}"
        assert not (s_val & s_test), f"Data leakage between val and test: {s_val & s_test}"

        return DatasetSplit(
            train_sequences=sorted(train_seqs),
            val_sequences=sorted(val_seqs),
            test_sequences=sorted(test_seqs),
        )

    @staticmethod
    def generate_yolo_yaml(
        dataset_root: Union[str, Path],
        taxonomy: ClassTaxonomyType = ClassTaxonomyType.FOUR_CLASS,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate YOLO-compatible dataset.yaml configuration file."""
        root = Path(dataset_root).resolve()
        class_names = ClassTaxonomy.get_names(taxonomy)

        yolo_config = {
            "path": root.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {int(k): v for k, v in class_names.items()},
        }

        yaml_content = yaml.dump(yolo_config, sort_keys=False, default_flow_style=False)

        if output_path is not None:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(yaml_content, encoding="utf-8")

        return yaml_content

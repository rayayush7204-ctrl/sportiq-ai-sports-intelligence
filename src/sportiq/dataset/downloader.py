"""Dataset downloader helper and repository verification for SoccerNet tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SoccerNetDownloaderHelper:
    """Manages verified interaction with SoccerNet tracking dataset repositories."""

    TRACKING_2023_REPO = "SoccerNet/SN-Tracking-2023"
    RAW_VIDEO_REPO = "SoccerNet/SoccerNet-Tracking-RAW-Video"

    @classmethod
    def inspect_remote_inventory(cls) -> Dict[str, Any]:
        """Inspect public Hugging Face repository metadata for SoccerNet/SN-Tracking-2023.

        Returns:
            Dict[str, Any]: Remote repository files, sizes, and access requirements.
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            dataset_info = api.dataset_info(cls.TRACKING_2023_REPO, files_metadata=True)

            files_info: List[Dict[str, Any]] = []
            for s in dataset_info.siblings:
                size_mb = round((s.size or 0) / (1024 * 1024), 2)
                files_info.append({
                    "filename": s.rfilename,
                    "size_bytes": s.size,
                    "size_mb": size_mb,
                })

            return {
                "repository": cls.TRACKING_2023_REPO,
                "status": "accessible",
                "gated": dataset_info.gated,
                "files_count": len(files_info),
                "files": files_info,
                "raw_video_repository": cls.RAW_VIDEO_REPO,
                "raw_video_gated": True,
                "raw_video_policy": (
                    "Raw broadcast video clips require an approved NDA agreement with SoccerNet. "
                    "Public redistribution of raw match video footage is strictly prohibited."
                ),
            }
        except ImportError:
            return {
                "repository": cls.TRACKING_2023_REPO,
                "status": "huggingface_hub_not_installed",
                "error": "Install huggingface_hub to query remote repositories.",
            }
        except Exception as e:
            return {
                "repository": cls.TRACKING_2023_REPO,
                "status": "error",
                "error": str(e),
            }

    @classmethod
    def verify_local_sequence_directory(cls, directory_path: Union[str, Path]) -> Dict[str, Any]:
        """Scan a local directory to verify whether it contains valid MOT sequences.

        Args:
            directory_path: Directory to inspect (e.g. data/raw/soccernet).

        Returns:
            Dict[str, Any]: Discovered sequences, missing files, and readiness status.
        """
        p = Path(directory_path).resolve()
        if not p.exists() or not p.is_dir():
            return {
                "path": str(p),
                "exists": False,
                "valid_sequences": [],
                "sequence_count": 0,
            }

        valid_sequences = []
        # Check subdirectories for seqinfo.ini and gt.txt
        for child in sorted(p.iterdir()):
            if child.is_dir():
                has_seqinfo = (child / "seqinfo.ini").exists()
                has_gt = (child / "gt" / "gt.txt").exists() or (child / "gt.txt").exists()
                if has_seqinfo and has_gt:
                    valid_sequences.append(child.name)

        return {
            "path": str(p),
            "exists": True,
            "valid_sequences": valid_sequences,
            "sequence_count": len(valid_sequences),
        }

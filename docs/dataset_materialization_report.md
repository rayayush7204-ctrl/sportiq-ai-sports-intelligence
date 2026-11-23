# SPORTIQ — PHASE 3, STEP 2B: DATASET MATERIALIZATION & STORAGE AUDIT REPORT

> **Document Status:** Storage Audit Complete · Selective Image Transfer Awaiting Budget Authorization

---

## 1. Executive Summary & Storage Audit Finding

During Phase 3 Step 2B, the remote public archives for SoccerNet Tracking 2023 (`SoccerNet/SN-Tracking-2023`) were audited at the ZIP central directory level via HTTP byte-range queries.

### Critical Budget Check
* **Pre-Approved Acquisition Budget:** **6.00 GiB**
* **Local Drive C: Free Space:** **60.97 GiB** (out of 256.73 GiB, > 20% required headroom satisfied)
* **Actual Image Volume for All 6 Games (79,500 frames):**
  * **Compressed Size:** **17.00 GiB** (17,406.7 MiB)
  * **Uncompressed Size:** **17.04 GiB** (17,452.8 MiB)
* **Budget Decision:** **TRANSFER HALTED AT AUDIT GATEWAY.** Full extraction of all 6 matches requires **17.04 GiB**, which exceeds the pre-approved **6.00 GiB** budget by **11.04 GiB** (~2.84x budget). In accordance with Step 2B safety rules, execution has stopped to present the exact storage figures and obtain user authorization before transferring data.

---

## 2. Remote ZIP Archive & Member Verification

| Archive | Remote Repository | Archive Size | HTTP Range Supported | Image Member Path Structure | Image File Format | Sequence Verification |
|:---|:---|:---|:---|:---|:---|:---|
| **`train.zip`** | `SoccerNet/SN-Tracking-2023` | 9,575,523,385 B (9.13 GB) | **Yes (`206 Partial Content`)** | `train/<sequence>/img1/<frame_id>.jpg` | JPEG (1920x1080 @ 25 FPS) | 57 / 57 verified present |
| **`test.zip`** | `SoccerNet/SN-Tracking-2023` | 8,711,601,474 B (8.31 GB) | **Yes (`206 Partial Content`)** | `test/<sequence>/img1/<frame_id>.jpg` | JPEG (1920x1080 @ 25 FPS) | 49 / 49 verified present |

### Member Path & Format Integrity
* **Path Traversal Protection:** Implemented `ZipPathSafetyValidator` blocking path traversal (`../`), leading slashes, drive letters, and unauthorized directory prefixes.
* **Image Naming Agreement:** Remote image files strictly follow `000001.jpg` to `000750.jpg` per sequence, agreeing 100% with `seqinfo.ini` (`seqLength=750`, `imDir=img1`, `imExt=.jpg`, `imWidth=1920`, `imHeight=1080`, `frameRate=25.0`).

---

## 3. Per-Match & Per-Split Storage Breakdown

The fixed internal project partitions group all 106 sequences into 6 mutually disjoint match groups:

| Split | Game ID | Sequences | Expected Frames | Annotations | Compressed Size | Uncompressed Size | Tracking Continuity Status |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **TRAIN** | `gameID=4` | 18 | 13,500 | 254,043 | 2.74 GiB (2,800.8 MiB) | 2.74 GiB (2,805.8 MiB) | Contiguous 25 FPS |
| **TRAIN** | `gameID=6` | 19 | 14,250 | 236,442 | 2.53 GiB (2,586.5 MiB) | 2.53 GiB (2,595.0 MiB) | Contiguous 25 FPS |
| **TRAIN** | `gameID=9` | 20 | 15,000 | 242,517 | 3.64 GiB (3,726.2 MiB) | 3.65 GiB (3,734.8 MiB) | Contiguous 25 FPS |
| *TRAIN Subtotal* | *3 Matches* | *57* | *42,750* | *733,002* | *8.91 GiB (9,113.5 MiB)* | *8.92 GiB (9,135.6 MiB)* | *Full contiguous* |
| **VAL** | `gameID=8` | 19 | 14,250 | 193,955 | 3.31 GiB (3,384.6 MiB) | 3.31 GiB (3,392.3 MiB) | Contiguous 25 FPS |
| *VAL Subtotal* | *1 Match* | *19* | *14,250* | *193,955* | *3.31 GiB (3,384.6 MiB)* | *3.31 GiB (3,392.3 MiB)* | *Full contiguous* |
| **TEST** | `gameID=7` | 16 | 12,000 | 182,533 | 2.78 GiB (2,846.6 MiB) | 2.79 GiB (2,854.0 MiB) | Contiguous 25 FPS |
| **TEST** | `gameID=11` | 14 | 10,500 | 188,059 | 2.01 GiB (2,061.8 MiB) | 2.02 GiB (2,070.9 MiB) | Contiguous 25 FPS |
| *TEST Subtotal* | *2 Matches* | *30* | *22,500* | *370,592* | *4.79 GiB (4,908.4 MiB)* | *4.81 GiB (4,924.9 MiB)* | *Full contiguous* |
| **TOTAL** | **6 Matches** | **106** | **79,500** | **1,297,549** | **17.00 GiB (17,406.7 MiB)** | **17.04 GiB (17,452.8 MiB)** | **100% Contiguous** |

---

## 4. Class Distribution by Split (Four-Class Target Taxonomy)

All 1,297,549 annotations across the proposed partitions are mapped to the four target detection categories (`staff` is cleanly isolated and excluded from detection):

| Category | Class ID | TRAIN (Games 4, 6, 9) | VAL (Game 8) | TEST (Games 7, 11) | Total Annotations |
|:---|:---|:---|:---|:---|:---|
| **Player** | `0` | 604,972 (82.5%) | 157,396 (81.1%) | 307,293 (82.9%) | **1,069,661 (82.4%)** |
| **Goalkeeper** | `1` | 24,107 (3.3%) | 6,967 (3.6%) | 12,492 (3.4%) | **43,566 (3.4%)** |
| **Referee** | `2` | 61,545 (8.4%) | 15,378 (7.9%) | 29,579 (8.0%) | **106,502 (8.2%)** |
| **Ball** | `3` | 40,931 (5.6%) | 13,582 (7.0%) | 20,547 (5.5%) | **75,060 (5.8%)** |
| **Staff (Unmapped)** | `None` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** |
| **Total** | — | **733,002** | **193,955** | **370,592** | **1,297,549** |

---

## 5. Storage Headroom & Acquisition Options

Drive C: has **60.97 GiB free**. Transferring 17.04 GiB will leave **43.93 GiB free**, retaining a **17.1%** free-space margin on the 256.73 GiB drive (well exceeding standard operational minimums, with 20% headroom being 51.3 GiB).

The following options are available for user authorization:

1. **Option A: Full 6-Match Acquisition (Recommended for Phase 3 Detection + Phase 4 Tracking):**
   * **Acquisition Size:** **17.04 GiB** (~79,500 contiguous frames)
   * **Result:** All 6 matches (Train, Val, Test) fully materialized with contiguous frames, establishing complete tracking continuity for Phase 4 ByteTrack / BoT-SORT without redundant re-downloads.
2. **Option B: Compact Detection-Only Train + Val Acquisition (Within ~6 GiB Budget):**
   * **Acquisition Size:** **5.84 GiB** (Games 6 [2.53 GiB] + Game 8 [3.31 GiB])
   * **Result:** Enables independent match-level train and validation under the 6 GiB limit, but defers test split image materialization.
3. **Option C: Uniform Temporal Subsampling (Detection-Only, e.g. 5 FPS / 1-in-5 Frames):**
   * **Acquisition Size:** **~3.41 GiB** (~15,900 frames across all 6 matches)
   * **Result:** High diversity across all 6 matches under 4 GiB, but **not tracking-ready** due to temporal gaps.

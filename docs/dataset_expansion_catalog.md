# SPORTIQ — SOCCERNET MULTI-MATCH DATASET CATALOG & AUDIT REPORT

> **Audit Scope:** SoccerNet Tracking (SN-Tracking-2023) public annotations and metadata.

## 1. Remote Annotation Repository Inventory

| Asset / Split | Remote Location | Size | Type | Ground Truth Status |
|:---|:---|:---|:---|:---|
| **train.zip** | `SoccerNet/SN-Tracking-2023` | 9,575 MB (~9.13 GB) | Split Archive | Public (57 sequences, 100% gt.txt) |
| **test.zip** | `SoccerNet/SN-Tracking-2023` | 8,711 MB (~8.31 GB) | Split Archive | Public (49 sequences, 100% gt.txt) |
| **challenge2023.zip** | `SoccerNet/SN-Tracking-2023` | 5,311 MB (~5.06 GB) | Challenge Archive | Labels Withheld (eval server) |
| **single_camera_games_public.zip** | `SoccerNet/SoccerNet-Tracking-RAW-Video` | 88,734 MB (~88.73 GB) | Raw Broadcast Video | Gated (NDA required) |

## 2. Catalog Overview & Summary Metrics

- **Total Discovered Annotated Sequences:** 106
- **Total Ground-Truth Annotations:** 1,297,549
- **Total Unique Ground-Truth Matches (`game_id`):** 6
- **Existing Known Match:** `gameID=7` (16 sequences)
- **Candidate Independent Matches Beyond Game 7:** `gameID=4`, `gameID=6`, `gameID=8`, `gameID=9`, `gameID=11`

## 3. Match Group Aggregations

| Match Group | Official Split(s) | Sequences | Total Annotations | Player | Goalkeeper | Referee | Ball | Staff | Local Frames |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Game 4** | train | 18 | 254,043 | 212,985 | 6,881 | 20,395 | 12,599 | 0 | 0 (Missing) |
| **Game 6** | train | 19 | 236,442 | 194,920 | 8,501 | 19,176 | 13,845 | 0 | 0 (Missing) |
| **Game 7** | test | 16 | 182,533 | 151,458 | 6,637 | 13,795 | 10,643 | 0 | 3 frames |
| **Game 8** | test | 19 | 193,955 | 157,396 | 6,967 | 15,378 | 13,582 | 0 | 0 (Missing) |
| **Game 9** | train | 20 | 242,517 | 197,067 | 8,725 | 21,975 | 14,487 | 0 | 0 (Missing) |
| **Game 11** | test | 14 | 188,059 | 155,835 | 5,855 | 15,784 | 9,904 | 0 | 0 (Missing) |

## 4. Sequence-to-Match Mapping Table

| Sequence Name | Split | Game ID | Action Class | Length (Frames) | FPS | Resolution | Annotations | Unique Tracks | Local Frames |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `SNMOT-060` | train | Game 4 | Kick-off | 750 | 25 | 1920x1080 | 13,540 | 26 | Missing |
| `SNMOT-061` | train | Game 4 | Shots on target | 750 | 25 | 1920x1080 | 11,189 | 27 | Missing |
| `SNMOT-062` | train | Game 4 | Foul | 750 | 25 | 1920x1080 | 13,383 | 24 | Missing |
| `SNMOT-063` | train | Game 4 | Offside | 750 | 25 | 1920x1080 | 12,223 | 25 | Missing |
| `SNMOT-064` | train | Game 4 | Shots off target | 750 | 25 | 1920x1080 | 16,225 | 24 | Missing |
| `SNMOT-065` | train | Game 4 | Clearance | 750 | 25 | 1920x1080 | 13,690 | 26 | Missing |
| `SNMOT-066` | train | Game 4 | Direct free-kick | 750 | 25 | 1920x1080 | 15,746 | 25 | Missing |
| `SNMOT-067` | train | Game 4 | Corner | 750 | 25 | 1920x1080 | 14,739 | 26 | Missing |
| `SNMOT-068` | train | Game 4 | Yellow card | 750 | 25 | 1920x1080 | 13,396 | 24 | Missing |
| `SNMOT-069` | train | Game 4 | Kick-off | 750 | 25 | 1920x1080 | 12,828 | 24 | Missing |
| `SNMOT-070` | train | Game 4 | Shots on target | 750 | 25 | 1920x1080 | 11,346 | 25 | Missing |
| `SNMOT-071` | train | Game 4 | Direct free-kick | 750 | 25 | 1920x1080 | 15,002 | 22 | Missing |
| `SNMOT-072` | train | Game 4 | Yellow card | 750 | 25 | 1920x1080 | 13,043 | 24 | Missing |
| `SNMOT-073` | train | Game 4 | Shots off target | 750 | 25 | 1920x1080 | 14,171 | 25 | Missing |
| `SNMOT-074` | train | Game 4 | Clearance | 750 | 25 | 1920x1080 | 15,852 | 25 | Missing |
| `SNMOT-075` | train | Game 4 | Corner | 750 | 25 | 1920x1080 | 14,519 | 24 | Missing |
| `SNMOT-076` | train | Game 4 | Substitution | 750 | 25 | 1920x1080 | 18,709 | 29 | Missing |
| `SNMOT-077` | train | Game 4 | Foul | 750 | 25 | 1920x1080 | 14,442 | 24 | Missing |
| `SNMOT-097` | train | Game 6 | Shots on target | 750 | 25 | 1920x1080 | 12,832 | 22 | Missing |
| `SNMOT-098` | train | Game 6 | Clearance | 750 | 25 | 1920x1080 | 9,639 | 26 | Missing |
| `SNMOT-099` | train | Game 6 | Yellow card | 750 | 25 | 1920x1080 | 10,392 | 25 | Missing |
| `SNMOT-100` | train | Game 6 | Direct free-kick | 750 | 25 | 1920x1080 | 13,205 | 26 | Missing |
| `SNMOT-101` | train | Game 6 | Goal | 750 | 25 | 1920x1080 | 8,767 | 25 | Missing |
| `SNMOT-102` | train | Game 6 | Kick-off | 750 | 25 | 1920x1080 | 13,228 | 26 | Missing |
| `SNMOT-103` | train | Game 6 | Corner | 750 | 25 | 1920x1080 | 15,267 | 24 | Missing |
| `SNMOT-104` | train | Game 6 | Goal | 750 | 25 | 1920x1080 | 14,706 | 23 | Missing |
| `SNMOT-105` | train | Game 6 | Shots off target | 750 | 25 | 1920x1080 | 13,085 | 26 | Missing |
| `SNMOT-106` | train | Game 6 | Goal | 750 | 25 | 1920x1080 | 12,292 | 25 | Missing |
| `SNMOT-107` | train | Game 6 | Kick-off | 750 | 25 | 1920x1080 | 12,571 | 25 | Missing |
| `SNMOT-108` | train | Game 6 | Clearance | 750 | 25 | 1920x1080 | 11,248 | 26 | Missing |
| `SNMOT-109` | train | Game 6 | Direct free-kick | 750 | 25 | 1920x1080 | 13,736 | 24 | Missing |
| `SNMOT-110` | train | Game 6 | Corner | 750 | 25 | 1920x1080 | 13,356 | 27 | Missing |
| `SNMOT-111` | train | Game 6 | Yellow card | 750 | 25 | 1920x1080 | 11,444 | 28 | Missing |
| `SNMOT-112` | train | Game 6 | Shots on target | 750 | 25 | 1920x1080 | 11,840 | 26 | Missing |
| `SNMOT-113` | train | Game 6 | Substitution | 750 | 25 | 1920x1080 | 15,760 | 31 | Missing |
| `SNMOT-114` | train | Game 6 | Foul | 750 | 25 | 1920x1080 | 9,919 | 24 | Missing |
| `SNMOT-115` | train | Game 6 | Shots off target | 750 | 25 | 1920x1080 | 13,155 | 23 | Missing |
| `SNMOT-116` | test | Game 7 | Corner | 750 | 25 | 1920x1080 | 11,460 | 27 | 3 |
| `SNMOT-117` | test | Game 7 | Offside | 750 | 25 | 1920x1080 | 9,171 | 31 | Missing |
| `SNMOT-118` | test | Game 7 | Shots off target | 750 | 25 | 1920x1080 | 10,919 | 25 | Missing |
| `SNMOT-119` | test | Game 7 | Clearance | 750 | 25 | 1920x1080 | 8,543 | 26 | Missing |
| `SNMOT-120` | test | Game 7 | Foul | 750 | 25 | 1920x1080 | 11,695 | 27 | Missing |
| `SNMOT-121` | test | Game 7 | Yellow card | 750 | 25 | 1920x1080 | 11,289 | 23 | Missing |
| `SNMOT-122` | test | Game 7 | Direct free-kick | 750 | 25 | 1920x1080 | 11,649 | 26 | Missing |
| `SNMOT-123` | test | Game 7 | Shots on target | 750 | 25 | 1920x1080 | 12,391 | 25 | Missing |
| `SNMOT-124` | test | Game 7 | Kick-off | 750 | 25 | 1920x1080 | 9,702 | 25 | Missing |
| `SNMOT-125` | test | Game 7 | Corner | 750 | 25 | 1920x1080 | 9,403 | 26 | Missing |
| `SNMOT-126` | test | Game 7 | Clearance | 750 | 25 | 1920x1080 | 10,569 | 25 | Missing |
| `SNMOT-127` | test | Game 7 | Substitution | 750 | 25 | 1920x1080 | 14,561 | 22 | Missing |
| `SNMOT-128` | test | Game 7 | Yellow card | 750 | 25 | 1920x1080 | 13,932 | 24 | Missing |
| `SNMOT-129` | test | Game 7 | Shots on target | 750 | 25 | 1920x1080 | 12,135 | 25 | Missing |
| `SNMOT-130` | test | Game 7 | Shots off target | 750 | 25 | 1920x1080 | 11,263 | 21 | Missing |
| `SNMOT-131` | test | Game 7 | Direct free-kick | 750 | 25 | 1920x1080 | 13,851 | 23 | Missing |
| `SNMOT-132` | test | Game 8 | Kick-off | 750 | 25 | 1920x1080 | 11,417 | 25 | Missing |
| `SNMOT-133` | test | Game 8 | Yellow card | 750 | 25 | 1920x1080 | 12,448 | 29 | Missing |
| `SNMOT-134` | test | Game 8 | Offside | 750 | 25 | 1920x1080 | 9,092 | 23 | Missing |
| `SNMOT-135` | test | Game 8 | Direct free-kick | 750 | 25 | 1920x1080 | 12,542 | 24 | Missing |
| `SNMOT-136` | test | Game 8 | Shots on target | 750 | 25 | 1920x1080 | 9,291 | 25 | Missing |
| `SNMOT-137` | test | Game 8 | Clearance | 750 | 25 | 1920x1080 | 9,005 | 25 | Missing |
| `SNMOT-138` | test | Game 8 | Foul | 750 | 25 | 1920x1080 | 12,511 | 25 | Missing |
| `SNMOT-139` | test | Game 8 | Shots off target | 750 | 25 | 1920x1080 | 10,537 | 21 | Missing |
| `SNMOT-140` | test | Game 8 | Corner | 750 | 25 | 1920x1080 | 13,306 | 23 | Missing |
| `SNMOT-141` | test | Game 8 | Foul | 750 | 25 | 1920x1080 | 10,492 | 24 | Missing |
| `SNMOT-142` | test | Game 8 | Shots on target | 750 | 25 | 1920x1080 | 11,053 | 25 | Missing |
| `SNMOT-143` | test | Game 8 | Offside | 750 | 25 | 1920x1080 | 7,588 | 24 | Missing |
| `SNMOT-144` | test | Game 8 | Penalty | 750 | 25 | 1920x1080 | 9,938 | 25 | Missing |
| `SNMOT-145` | test | Game 8 | Corner | 750 | 25 | 1920x1080 | 9,584 | 25 | Missing |
| `SNMOT-146` | test | Game 8 | Shots off target | 750 | 25 | 1920x1080 | 7,453 | 22 | Missing |
| `SNMOT-147` | test | Game 8 | Clearance | 750 | 25 | 1920x1080 | 6,272 | 27 | Missing |
| `SNMOT-148` | test | Game 8 | Goal | 750 | 25 | 1920x1080 | 10,232 | 23 | Missing |
| `SNMOT-149` | test | Game 8 | Kick-off | 750 | 25 | 1920x1080 | 10,419 | 24 | Missing |
| `SNMOT-150` | test | Game 8 | Substitution | 750 | 25 | 1920x1080 | 10,775 | 29 | Missing |
| `SNMOT-151` | train | Game 9 | Kick-off | 750 | 25 | 1920x1080 | 10,626 | 25 | Missing |
| `SNMOT-152` | train | Game 9 | Offside | 750 | 25 | 1920x1080 | 15,950 | 22 | Missing |
| `SNMOT-153` | train | Game 9 | Goal | 750 | 25 | 1920x1080 | 10,079 | 28 | Missing |
| `SNMOT-154` | train | Game 9 | Corner | 750 | 25 | 1920x1080 | 12,977 | 24 | Missing |
| `SNMOT-155` | train | Game 9 | Shots off target | 750 | 25 | 1920x1080 | 11,048 | 24 | Missing |
| `SNMOT-156` | train | Game 9 | Direct free-kick | 750 | 25 | 1920x1080 | 12,728 | 24 | Missing |
| `SNMOT-157` | train | Game 9 | Clearance | 750 | 25 | 1920x1080 | 10,443 | 24 | Missing |
| `SNMOT-158` | train | Game 9 | Foul | 750 | 25 | 1920x1080 | 10,586 | 26 | Missing |
| `SNMOT-159` | train | Game 9 | Shots on target | 750 | 25 | 1920x1080 | 12,550 | 23 | Missing |
| `SNMOT-160` | train | Game 9 | Kick-off | 750 | 25 | 1920x1080 | 10,705 | 25 | Missing |
| `SNMOT-161` | train | Game 9 | Shots off target | 750 | 25 | 1920x1080 | 10,953 | 25 | Missing |
| `SNMOT-162` | train | Game 9 | Direct free-kick | 750 | 25 | 1920x1080 | 13,517 | 25 | Missing |
| `SNMOT-163` | train | Game 9 | Offside | 750 | 25 | 1920x1080 | 10,537 | 25 | Missing |
| `SNMOT-164` | train | Game 9 | Foul | 750 | 25 | 1920x1080 | 13,781 | 25 | Missing |
| `SNMOT-165` | train | Game 9 | Goal | 750 | 25 | 1920x1080 | 11,525 | 26 | Missing |
| `SNMOT-166` | train | Game 9 | Substitution | 750 | 25 | 1920x1080 | 13,407 | 26 | Missing |
| `SNMOT-167` | train | Game 9 | Corner | 750 | 25 | 1920x1080 | 12,924 | 25 | Missing |
| `SNMOT-168` | train | Game 9 | Yellow card | 750 | 25 | 1920x1080 | 13,772 | 23 | Missing |
| `SNMOT-169` | train | Game 9 | Clearance | 750 | 25 | 1920x1080 | 11,752 | 26 | Missing |
| `SNMOT-170` | train | Game 9 | Shots on target | 750 | 25 | 1920x1080 | 12,657 | 25 | Missing |
| `SNMOT-187` | test | Game 11 | Corner | 750 | 25 | 1920x1080 | 14,711 | 24 | Missing |
| `SNMOT-188` | test | Game 11 | Substitution | 750 | 25 | 1920x1080 | 17,865 | 27 | Missing |
| `SNMOT-189` | test | Game 11 | Foul | 750 | 25 | 1920x1080 | 12,817 | 25 | Missing |
| `SNMOT-190` | test | Game 11 | Shots off target | 750 | 25 | 1920x1080 | 11,628 | 24 | Missing |
| `SNMOT-191` | test | Game 11 | Offside | 750 | 25 | 1920x1080 | 12,217 | 24 | Missing |
| `SNMOT-192` | test | Game 11 | Yellow card | 750 | 25 | 1920x1080 | 11,607 | 26 | Missing |
| `SNMOT-193` | test | Game 11 | Direct free-kick | 750 | 25 | 1920x1080 | 12,334 | 26 | Missing |
| `SNMOT-194` | test | Game 11 | Clearance | 750 | 25 | 1920x1080 | 15,156 | 24 | Missing |
| `SNMOT-195` | test | Game 11 | Shots off target | 750 | 25 | 1920x1080 | 13,452 | 22 | Missing |
| `SNMOT-196` | test | Game 11 | Clearance | 750 | 25 | 1920x1080 | 11,401 | 24 | Missing |
| `SNMOT-197` | test | Game 11 | Foul | 750 | 25 | 1920x1080 | 13,862 | 24 | Missing |
| `SNMOT-198` | test | Game 11 | Direct free-kick | 750 | 25 | 1920x1080 | 14,717 | 27 | Missing |
| `SNMOT-199` | test | Game 11 | Corner | 750 | 25 | 1920x1080 | 13,836 | 25 | Missing |
| `SNMOT-200` | test | Game 11 | Shots on target | 750 | 25 | 1920x1080 | 12,456 | 20 | Missing |

## 5. Candidate Independent Matches Beyond Game 7

To satisfy strict match-level splitting without spatial/temporal data leakage, candidate matches must possess established distinct game IDs:

### Candidate Match: Game 4
- **Split:** train
- **Sequences (18):** `SNMOT-060`, `SNMOT-061`, `SNMOT-062`, `SNMOT-063`, `SNMOT-064`, `SNMOT-065`, `SNMOT-066`, `SNMOT-067`, `SNMOT-068`, `SNMOT-069`, `SNMOT-070`, `SNMOT-071`, `SNMOT-072`, `SNMOT-073`, `SNMOT-074`, `SNMOT-075`, `SNMOT-076`, `SNMOT-077`
- **Total Annotations:** 254,043
- **Class Distribution:** Player: 212,985, Goalkeeper: 6,881, Referee: 20,395, Ball: 12,599
- **Local Frame Status:** Missing on disk

### Candidate Match: Game 6
- **Split:** train
- **Sequences (19):** `SNMOT-097`, `SNMOT-098`, `SNMOT-099`, `SNMOT-100`, `SNMOT-101`, `SNMOT-102`, `SNMOT-103`, `SNMOT-104`, `SNMOT-105`, `SNMOT-106`, `SNMOT-107`, `SNMOT-108`, `SNMOT-109`, `SNMOT-110`, `SNMOT-111`, `SNMOT-112`, `SNMOT-113`, `SNMOT-114`, `SNMOT-115`
- **Total Annotations:** 236,442
- **Class Distribution:** Player: 194,920, Goalkeeper: 8,501, Referee: 19,176, Ball: 13,845
- **Local Frame Status:** Missing on disk

### Candidate Match: Game 8
- **Split:** test
- **Sequences (19):** `SNMOT-132`, `SNMOT-133`, `SNMOT-134`, `SNMOT-135`, `SNMOT-136`, `SNMOT-137`, `SNMOT-138`, `SNMOT-139`, `SNMOT-140`, `SNMOT-141`, `SNMOT-142`, `SNMOT-143`, `SNMOT-144`, `SNMOT-145`, `SNMOT-146`, `SNMOT-147`, `SNMOT-148`, `SNMOT-149`, `SNMOT-150`
- **Total Annotations:** 193,955
- **Class Distribution:** Player: 157,396, Goalkeeper: 6,967, Referee: 15,378, Ball: 13,582
- **Local Frame Status:** Missing on disk

### Candidate Match: Game 9
- **Split:** train
- **Sequences (20):** `SNMOT-151`, `SNMOT-152`, `SNMOT-153`, `SNMOT-154`, `SNMOT-155`, `SNMOT-156`, `SNMOT-157`, `SNMOT-158`, `SNMOT-159`, `SNMOT-160`, `SNMOT-161`, `SNMOT-162`, `SNMOT-163`, `SNMOT-164`, `SNMOT-165`, `SNMOT-166`, `SNMOT-167`, `SNMOT-168`, `SNMOT-169`, `SNMOT-170`
- **Total Annotations:** 242,517
- **Class Distribution:** Player: 197,067, Goalkeeper: 8,725, Referee: 21,975, Ball: 14,487
- **Local Frame Status:** Missing on disk

### Candidate Match: Game 11
- **Split:** test
- **Sequences (14):** `SNMOT-187`, `SNMOT-188`, `SNMOT-189`, `SNMOT-190`, `SNMOT-191`, `SNMOT-192`, `SNMOT-193`, `SNMOT-194`, `SNMOT-195`, `SNMOT-196`, `SNMOT-197`, `SNMOT-198`, `SNMOT-199`, `SNMOT-200`
- **Total Annotations:** 188,059
- **Class Distribution:** Player: 155,835, Goalkeeper: 5,855, Referee: 15,784, Ball: 9,904
- **Local Frame Status:** Missing on disk

## 6. Annotation and Image Availability Gaps

1. **Annotations Status (100% Complete):** All 106 sequences in the train (57) and test (49) splits have verified `gt.txt`, `gameinfo.ini`, and `seqinfo.ini` files extracted and cataloged.
2. **Image Frames Status (Gap Identified):** Currently, only sequence `SNMOT-116` has 3 extracted sample frames locally (`000001.jpg`, `000002.jpg`, `000003.jpg`). The remaining 105 sequences lack extracted image frames locally.
3. **Image Acquisition Source Options:**
   - **Option A (Pre-extracted Frames):** Extract image frames directly from public `train.zip` and `test.zip` in `SoccerNet/SN-Tracking-2023`. Does not require NDA video access.
   - **Option B (Raw Video Ingestion):** Download raw match broadcasts from gated `SoccerNet/SoccerNet-Tracking-RAW-Video` (~88.7 GB archive) and sample frames using Phase 1 `FrameExtractor`.

## 7. Storage Implications & Access Safeguards

- **Remote Raw Video Archive:** `single_camera_games_public.zip` is ~88.73 GB compressed. Decompressed full match broadcasts require approximately 120–150 GB of storage.
- **Gated Access Requirement:** `SoccerNet-Tracking-RAW-Video` is gated on Hugging Face and requires manual approval under a SoccerNet academic research agreement.
- **Storage Efficiency Recommendation:** Extracting only the required frame images for candidate matches from `SN-Tracking-2023` saves downloading and decompressing the 88.7 GB raw video archive.

## 8. Custom Four-Class Training Readiness Decision

> **DECISION: STRUCTURALLY RESOLVED AT METADATA LEVEL · OPERATIONAL EXECUTION BLOCKED ON FRAME IMAGES**

- **Metadata & Partitioning:** RESOLVED. Six distinct matches (Games 4, 6, 7, 8, 9, 11) provide more than the minimum 3 required disjoint match groups for zero-leakage train/validation/test splits.
- **Physical Training Readiness:** BLOCKED pending materialization of JPEG frame images for the candidate match sequences.
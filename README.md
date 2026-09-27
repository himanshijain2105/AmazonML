# Amazon ML Challenge 2026 — Business Entity Resolution

**Team Dhurandhar**

Large-scale business entity resolution system developed for the
Amazon ML Challenge 2026.

## Final Result

**Leaderboard score: 0.934946**

## Problem

The system matches noisy business records from S2/S3 against the
deduplicated S1 reference dataset.

The data contains variations in:

- business names
- legal suffixes
- abbreviations
- spelling / typographical noise
- address formatting
- address ordering
- postal codes
- house numbers
- multilingual and non-Latin text

## Approach

```text
Raw records
    ↓
Name + address normalization
    ↓
Multiple blocking key families
    ↓
Bounded top-k candidate generation
    ↓
Pairwise feature generation
    ↓
LightGBM matching model
    ↓
Decision thresholds
    ↓
Competitor / one-to-one resolution
    ↓
Submission

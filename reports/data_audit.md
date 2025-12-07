# TikTok Raw Data Audit

Quick exploratory stats gathered via `python3` + standard library (`csv` module) to avoid extra dependencies.

## File
- Path: `data/raw/tiktok_merged_data_deduplicated.csv`
- Rows: **7,225**
- Columns: **14**

## Column Names
`video_id`, `author`, `description`, `likes`, `comments`, `shares`, `plays`, `hashtags`, `music`, `create_time`, `video_url`, `fetch_time`, `views`, `posted_time`

## Missingness Snapshot
| Column       | Missing Rows | Notes |
|--------------|--------------|-------|
| `fetch_time` | 7,218        | Likely scraped without capture timestamp; treat as optional. |
| `views`      | 7,218        | Duplicates of `plays`? needs clarification before modeling. |
| `posted_time`| 7,218        | Might require recomputing from `create_time`. |
| `hashtags`   | 2,075        | Empty strings mean organic/no-tag posts. |
| `description`|   692        | Expect some posts without captions; could backfill with hashtags. |
| `plays`      |     7        | Few missing engagement counts; drop or impute later. |
| `create_time`|     7        | Need timezone handling; drop if missing. |

## Sample Records (first 3)
1. Rodeo-themed clip with `686k` plays and hashtags `rodeotime, dalebrisby, jbmauney`.
2. Personal shoutout video (`1.2M` plays) with no hashtags.
3. Public interview snippet using `#publicinterview`/`#rizz`.

## Next Steps
1. Normalize `create_time` → canonical timezone; reconstruct `posted_time` if necessary.
2. Decide whether `views` duplicates `plays`; drop redundant column to simplify schema.
3. Treat empty `hashtags` as “no tags” category to keep feature pipeline straightforward.


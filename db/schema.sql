-- Schema for storing TikTok/Instagram post analytics.
-- Enable required extensions (run once per database)
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- The trigram extension speeds up fuzzy hashtag searches. Keep it commented
-- until the DBA confirms the extension is available.

-- Creators: one row per account we observe. We separate this from posts so
-- we can reuse creator metadata (e.g., follower count, timezone) across all
-- their videos.
CREATE TABLE IF NOT EXISTS creators (
    creator_id SERIAL PRIMARY KEY, -- internal surrogate key
    handle TEXT NOT NULL UNIQUE, -- @username scraped from the platform
    platform TEXT NOT NULL DEFAULT 'tiktok' CHECK (platform IN ('tiktok', 'instagram')), -- keep platform field explicit for future IG data
    follower_count BIGINT, -- optional: may be NULL if the scraper did not capture it
    home_timezone TEXT, -- e.g., 'America/Los_Angeles'; helps personalize posting windows
    created_at TIMESTAMPTZ DEFAULT NOW() -- auto timestamp for auditing
);

-- Music tracks: deduplicated catalog of audio clips.
-- A single track can be referenced by many posts so we normalize it here.
CREATE TABLE IF NOT EXISTS music_tracks (
    music_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL, -- track name as displayed on TikTok
    artist TEXT, -- may be empty for user-generated sounds
    platform TEXT CHECK (platform IN ('tiktok', 'instagram')), -- ensures we do not mix IG audio with TikTok
    is_original BOOLEAN DEFAULT FALSE, -- true when sound = “Original Audio”
    first_seen TIMESTAMPTZ, -- when the scraper first encountered this track
    last_seen TIMESTAMPTZ, -- last observation, useful for trend decay
    UNIQUE (title, COALESCE(artist, ''), COALESCE(platform, 'tiktok')) -- prevents duplicates created by NULL artist/platform combos
);

-- Posts: the central fact table describing each piece of content.
-- Ties together the creator, music, timestamps, and raw metadata.
CREATE TABLE IF NOT EXISTS posts (
    post_id BIGSERIAL PRIMARY KEY, -- surrogate key for joins
    video_id TEXT NOT NULL UNIQUE, -- TikTok video id (matches CSV column)
    creator_id INTEGER NOT NULL REFERENCES creators(creator_id) ON DELETE CASCADE, -- cascade deletes if the creator is purged
    music_id INTEGER REFERENCES music_tracks(music_id) ON DELETE SET NULL, -- allow posts to remain even if we drop duplicate tracks
    caption TEXT, -- raw caption/description text
    raw_hashtags TEXT, -- comma-separated list; normalized in post_hashtags
    video_url TEXT, -- direct URL for manual QA
    source_platform TEXT NOT NULL CHECK (source_platform IN ('tiktok', 'instagram')), -- keeps schema future-proof for Instagram data
    created_at TIMESTAMPTZ, -- original posting time parsed from API/scrape
    fetched_at TIMESTAMPTZ, -- when our pipeline grabbed the row (maps to CSV fetch_time)
    duration_seconds INTEGER, -- optional: not present in the CSV yet
    language TEXT, -- placeholder for future language detection
    inserted_at TIMESTAMPTZ DEFAULT NOW(), -- when the row entered the DB
    CONSTRAINT fetched_after_created CHECK (
        fetched_at IS NULL OR created_at IS NULL OR fetched_at >= created_at
    )
);

-- Hashtags master table to deduplicate and index hashtags efficiently.
CREATE TABLE IF NOT EXISTS hashtags (
    hashtag_id SERIAL PRIMARY KEY,
    tag TEXT NOT NULL UNIQUE -- store lowercase tag text without '#'
);

-- Many-to-many bridge associating posts with hashtags and their order.
CREATE TABLE IF NOT EXISTS post_hashtags (
    post_id BIGINT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE, -- remove relations when post disappears
    hashtag_id INTEGER NOT NULL REFERENCES hashtags(hashtag_id) ON DELETE CASCADE, -- remove relations when tag is deprecated
    position SMALLINT, -- 0-based index of the hashtag within the caption
    PRIMARY KEY (post_id, hashtag_id)
);

-- Engagement snapshot table stores the metrics we care about (likes, plays,
-- etc.) at different time offsets (1h, 3h, 24h). This enables us to compute
-- engagement velocity without rescraping the raw post.
CREATE TABLE IF NOT EXISTS engagement_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE, -- tie snapshot back to the canonical post
    hours_since_post NUMERIC(6,2) NOT NULL, -- example: 1.00, 3.00, 24.00
    likes BIGINT DEFAULT 0,
    comments BIGINT DEFAULT 0,
    shares BIGINT DEFAULT 0,
    plays BIGINT,
    views BIGINT,
    capture_time TIMESTAMPTZ NOT NULL, -- actual timestamp we scraped the metrics
    UNIQUE (post_id, hours_since_post)
);

-- Helpful indexes for common queries (time series charts, creator views, etc.).
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_creator ON posts(creator_id);
CREATE INDEX IF NOT EXISTS idx_posts_music ON posts(music_id);
CREATE INDEX IF NOT EXISTS idx_engagement_post_hours ON engagement_snapshots(post_id, hours_since_post);
CREATE INDEX IF NOT EXISTS idx_hashtag_tag_trgm ON hashtags USING gin (tag gin_trgm_ops);

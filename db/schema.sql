-- Schema for storing TikTok/Instagram post analytics.
-- Enable required extensions (run once per database)
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS creators (
    creator_id SERIAL PRIMARY KEY,
    handle TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL DEFAULT 'tiktok' CHECK (platform IN ('tiktok', 'instagram')),
    follower_count BIGINT,
    home_timezone TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS music_tracks (
    music_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT,
    platform TEXT CHECK (platform IN ('tiktok', 'instagram')),
    is_original BOOLEAN DEFAULT FALSE,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    UNIQUE (title, COALESCE(artist, ''), COALESCE(platform, 'tiktok'))
);

CREATE TABLE IF NOT EXISTS posts (
    post_id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL UNIQUE,
    creator_id INTEGER NOT NULL REFERENCES creators(creator_id) ON DELETE CASCADE,
    music_id INTEGER REFERENCES music_tracks(music_id) ON DELETE SET NULL,
    caption TEXT,
    raw_hashtags TEXT,
    video_url TEXT,
    source_platform TEXT NOT NULL CHECK (source_platform IN ('tiktok', 'instagram')),
    created_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    language TEXT,
    inserted_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fetched_after_created CHECK (
        fetched_at IS NULL OR created_at IS NULL OR fetched_at >= created_at
    )
);

CREATE TABLE IF NOT EXISTS hashtags (
    hashtag_id SERIAL PRIMARY KEY,
    tag TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS post_hashtags (
    post_id BIGINT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    hashtag_id INTEGER NOT NULL REFERENCES hashtags(hashtag_id) ON DELETE CASCADE,
    position SMALLINT,
    PRIMARY KEY (post_id, hashtag_id)
);

CREATE TABLE IF NOT EXISTS engagement_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    hours_since_post NUMERIC(6,2) NOT NULL,
    likes BIGINT DEFAULT 0,
    comments BIGINT DEFAULT 0,
    shares BIGINT DEFAULT 0,
    plays BIGINT,
    views BIGINT,
    capture_time TIMESTAMPTZ NOT NULL,
    UNIQUE (post_id, hours_since_post)
);

CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_creator ON posts(creator_id);
CREATE INDEX IF NOT EXISTS idx_posts_music ON posts(music_id);
CREATE INDEX IF NOT EXISTS idx_engagement_post_hours ON engagement_snapshots(post_id, hours_since_post);
CREATE INDEX IF NOT EXISTS idx_hashtag_tag_trgm ON hashtags USING gin (tag gin_trgm_ops);

-- init_db.sql
--
-- This script runs automatically when the PostgreSQL container starts
-- for the first time. It creates the pgvector extension and all tables
-- needed by the Fantasy Football MCP Server.
--
-- If you need to reset the database, run:
--   docker compose down -v && docker compose up -d
--
-- The tables serve three memory layers:
--   1. user_sessions / rosters / league_settings → Session & Persistent Memory
--   2. conversation_memory (with vector column)  → Semantic Memory (pgvector)

-- Enable the pgvector extension
-- This adds the 'vector' data type and similarity operators (<=>)
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Session & Persistent Memory ───────────────────────────

-- Tracks user sessions for continuity
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW(),
    preferences JSONB DEFAULT '{}'::jsonb
);

-- Stores player rosters per user
-- Each player is a separate row for flexible querying
CREATE TABLE IF NOT EXISTS rosters (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, player_name)
);

-- League configuration per user
CREATE TABLE IF NOT EXISTS league_settings (
    user_id TEXT PRIMARY KEY,
    league_name TEXT,
    scoring_format TEXT DEFAULT 'ppr',
    num_teams INTEGER DEFAULT 12,
    roster_slots JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ─── Semantic Memory (pgvector) ────────────────────────────

-- This is the key table that uses pgvector.
-- The 'embedding' column stores 384-dimensional vectors
-- (matching all-MiniLM-L6-v2 output dimensions).
-- 
-- When we search past conversations, we:
--   1. Convert the search query to a 384-dim embedding
--   2. Use the <=> operator to compute cosine distance
--   3. ORDER BY distance ASC (lowest distance = highest similarity)
--   4. Return the most relevant past conversations

CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    players_mentioned TEXT[] DEFAULT ARRAY[]::TEXT[],
    embedding vector(384),      -- pgvector column: 384-dim float vector
    created_at TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index for approximate nearest neighbor search
-- This makes vector search fast (O(sqrt(n)) instead of O(n))
-- 'lists = 100' means vectors are partitioned into 100 clusters
-- At query time, only nearby clusters are searched
CREATE INDEX IF NOT EXISTS idx_conversation_memory_embedding
ON conversation_memory
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Regular B-tree indexes for non-vector lookups
CREATE INDEX IF NOT EXISTS idx_rosters_user_id ON rosters(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_memory_user_id ON conversation_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);

-- Verify setup
DO $$
BEGIN
    RAISE NOTICE 'Fantasy Football MCP database initialized successfully!';
    RAISE NOTICE 'Tables: user_sessions, rosters, league_settings, conversation_memory';
    RAISE NOTICE 'pgvector extension enabled with IVFFlat indexing';
END $$;

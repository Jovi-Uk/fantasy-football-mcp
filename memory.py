"""
Memory Manager for the Fantasy Football MCP Server.

This module handles ALL persistent memory using PostgreSQL + pgvector.
It manages three distinct types of memory:

1. SESSION MEMORY (roster tracking):
   - Plain relational SQL in the 'rosters' and 'user_sessions' tables
   - Stores the user's current roster, league settings, preferences
   - Fast exact lookups by user_id and session_id
   - No vectors needed — this is structured data

2. PERSISTENT MEMORY (cross-session continuity):
   - Same tables as session memory, but keyed by user_id (not session_id)
   - When a user returns, their roster and preferences are still there
   - Enables the "Hey, I'm back — what lineup should I set?" experience

3. SEMANTIC MEMORY (conversation history via pgvector):
   - This is where pgvector becomes essential
   - Each conversation interaction is summarized and stored as a text embedding
   - When the agent needs to recall past advice, it performs a cosine similarity
     search against stored embeddings to find the most relevant past interactions
   - This is fundamentally the same RAG retrieval concept you use in Dify,
     but applied to the user's OWN conversation history instead of static docs

Architecture decisions:
   - We use asyncpg (async PostgreSQL driver) because MCP tools are async
   - Embeddings are generated using sentence-transformers (runs locally, no API key)
   - The embedding model (all-MiniLM-L6-v2) produces 384-dimensional vectors
   - We use IVFFlat indexing in pgvector for fast approximate nearest neighbor search
   - Connection pooling via asyncpg.Pool keeps database connections efficient

Database schema is defined in init_db.sql and created on first run.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import asyncpg
import numpy as np

logger = logging.getLogger("fantasy_football_mcp.memory")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Embedding dimension — must match your embedding model
# all-MiniLM-L6-v2 produces 384-dimensional vectors
# If you switch to OpenAI ada-002, change this to 1536
EMBEDDING_DIM = 384

# Default number of results for semantic search
DEFAULT_SEARCH_LIMIT = 5


# ─────────────────────────────────────────────────────────────
# Embedding Generator
# ─────────────────────────────────────────────────────────────

class EmbeddingGenerator:
    """
    Generates text embeddings for semantic memory storage and retrieval.
    
    We use sentence-transformers locally so:
      1. No API key needed (great for a class project)
      2. No per-request cost
      3. Works offline
      4. Fast enough for our use case (~50ms per embedding)
    
    The model (all-MiniLM-L6-v2) maps text to a 384-dimensional vector
    where semantically similar texts have high cosine similarity.
    
    Example:
      "Derrick Henry ankle injury" and "Henry's hurt ankle" 
      would have very similar embeddings (high cosine similarity)
      even though the exact words are different.
    """

    def __init__(self):
        self._model = None

    async def initialize(self):
        """
        Load the embedding model.
        
        This uses sentence-transformers which downloads the model on first run
        (~80MB). Subsequent runs use the cached version.
        
        If sentence-transformers isn't available, we fall back to a simple
        hashing-based approach (less accurate but zero dependencies).
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers embedding model")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback hash-based embeddings. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None

    def generate(self, text: str) -> List[float]:
        """
        Convert text to a vector embedding.
        
        The returned vector is what gets stored in pgvector's vector column
        and used for cosine similarity search.
        
        Args:
            text: Any text string (player advice, query, summary, etc.)
        
        Returns:
            List of floats — the embedding vector (384 dimensions)
        """
        if self._model is not None:
            # Use the real sentence-transformers model
            embedding = self._model.encode(text)
            return embedding.tolist()
        else:
            # Fallback: deterministic hash-based embedding
            # This won't capture semantic meaning as well, but it works
            # for testing the pgvector pipeline without extra dependencies
            return self._hash_embedding(text)

    @staticmethod
    def _hash_embedding(text: str) -> List[float]:
        """
        Fallback embedding using hash-based approach.
        
        This creates a pseudo-random but deterministic vector from text.
        Same text always produces the same vector.
        Not semantically meaningful, but tests the pgvector pipeline.
        """
        import hashlib
        np.random.seed(int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32))
        return np.random.randn(EMBEDDING_DIM).tolist()


# ─────────────────────────────────────────────────────────────
# Memory Manager
# ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Manages all persistent memory operations via PostgreSQL + pgvector.
    
    This is instantiated once during server lifespan and accessed by
    all memory-related MCP tools through the Context system.
    
    Lifecycle:
      1. Server starts → lifespan calls initialize() → connection pool created
      2. Tools call roster/memory methods → queries run against the pool
      3. Server stops → lifespan calls close() → pool cleaned up
    
    The connection pool (asyncpg.Pool) maintains multiple database connections
    so concurrent tool calls don't block each other.
    """

    def __init__(self, database_url: str):
        """
        Args:
            database_url: PostgreSQL connection string
                          e.g., "postgresql://postgres:password@localhost:5432/fantasy_bot"
        """
        self._database_url = database_url
        self._pool: Optional[asyncpg.Pool] = None
        self._embedder = EmbeddingGenerator()

    async def initialize(self):
        """
        Set up the database connection pool and ensure tables exist.
        
        Called during server lifespan startup. Creates:
          1. Connection pool (5-20 connections)
          2. pgvector extension (if not already installed)
          3. All required tables (if they don't exist)
          4. Embedding model loading
        """
        logger.info("Initializing memory manager...")
        
        # Create connection pool
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=2,      # Minimum connections to keep open
            max_size=10,     # Maximum connections allowed
            command_timeout=30
        )
        
        # Ensure database schema exists
        await self._create_tables()
        
        # Initialize embedding generator
        await self._embedder.initialize()
        
        logger.info("Memory manager initialized successfully")

    async def _create_tables(self):
        """
        Create all required database tables if they don't exist.
        
        This is idempotent — safe to call on every startup.
        The schema mirrors what's in init_db.sql but ensures it
        exists even if the SQL file wasn't run manually.
        """
        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # User sessions table — tracks who's connected and when
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_active TIMESTAMP DEFAULT NOW(),
                    preferences JSONB DEFAULT '{}'::jsonb
                );
            """)
            
            # Rosters table — stores player rosters per user
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rosters (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    position TEXT,
                    team TEXT,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, player_name)
                );
            """)
            
            # League settings table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS league_settings (
                    user_id TEXT PRIMARY KEY,
                    league_name TEXT,
                    scoring_format TEXT DEFAULT 'ppr',
                    num_teams INTEGER DEFAULT 12,
                    roster_slots JSONB DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Conversation memory table — THIS IS THE PGVECTOR TABLE
            # The 'embedding' column stores the vector representation
            # of each conversation summary for semantic search
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    players_mentioned TEXT[] DEFAULT ARRAY[]::TEXT[],
                    embedding vector({EMBEDDING_DIM}),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create index for fast vector similarity search
            # IVFFlat is an approximate nearest neighbor index
            # It partitions vectors into lists and only searches nearby lists
            # This makes search O(sqrt(n)) instead of O(n)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_conversation_memory_embedding
                ON conversation_memory
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Regular indexes for non-vector queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rosters_user_id ON rosters(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_memory_user_id 
                    ON conversation_memory(user_id);
            """)

    async def close(self):
        """Clean up database connections on server shutdown."""
        if self._pool:
            await self._pool.close()
            logger.info("Memory manager connection pool closed")

    # ─────────────────────────────────────────────────────────
    # Roster Operations (Session + Persistent Memory)
    # ─────────────────────────────────────────────────────────

    async def store_roster(
        self,
        user_id: str,
        players: List[str],
        scoring_format: str = "ppr",
        league_name: Optional[str] = None
    ) -> dict:
        """
        Store or replace a user's fantasy roster.
        
        This implements both session and persistent memory:
        - The roster is keyed by user_id, so it persists across sessions
        - Each player is stored as a separate row for easy querying
        - We use UPSERT (INSERT ON CONFLICT) to handle re-storing gracefully
        
        Args:
            user_id: Unique user identifier
            players: List of player names
            scoring_format: League scoring format
            league_name: Optional league name
            
        Returns:
            Dict with stored_count and any players that couldn't be resolved
        """
        async with self._pool.acquire() as conn:
            # Clear existing roster for this user (full replacement)
            await conn.execute(
                "DELETE FROM rosters WHERE user_id = $1", user_id
            )
            
            stored = []
            not_found = []
            
            for player_name in players:
                # We store the name as-is; position/team can be enriched later
                await conn.execute(
                    """
                    INSERT INTO rosters (user_id, player_name, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (user_id, player_name) DO UPDATE
                    SET updated_at = NOW()
                    """,
                    user_id, player_name
                )
                stored.append(player_name)
            
            # Store league settings if provided
            if league_name or scoring_format:
                await conn.execute(
                    """
                    INSERT INTO league_settings (user_id, league_name, scoring_format, updated_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (user_id) DO UPDATE
                    SET league_name = COALESCE($2, league_settings.league_name),
                        scoring_format = $3,
                        updated_at = NOW()
                    """,
                    user_id, league_name, scoring_format
                )
            
            return {
                "stored_count": len(stored),
                "players": stored,
                "not_found": not_found
            }

    async def get_roster(self, user_id: str) -> Optional[dict]:
        """
        Retrieve a user's stored roster and league settings.
        
        Returns None if the user has no stored roster.
        """
        async with self._pool.acquire() as conn:
            # Get roster players
            rows = await conn.fetch(
                """
                SELECT player_name, position, team, updated_at
                FROM rosters
                WHERE user_id = $1
                ORDER BY player_name
                """,
                user_id
            )
            
            if not rows:
                return None
            
            # Get league settings
            settings = await conn.fetchrow(
                "SELECT * FROM league_settings WHERE user_id = $1",
                user_id
            )
            
            return {
                "user_id": user_id,
                "players": [
                    {
                        "name": row["player_name"],
                        "position": row["position"],
                        "team": row["team"],
                        "updated_at": str(row["updated_at"])
                    }
                    for row in rows
                ],
                "league_settings": dict(settings) if settings else None,
                "player_count": len(rows)
            }

    async def update_roster(
        self,
        user_id: str,
        add_players: Optional[List[str]] = None,
        drop_players: Optional[List[str]] = None
    ) -> dict:
        """
        Add or drop players from an existing roster.
        
        This enables the "I just picked up Player X from waivers" flow
        without having to re-submit the entire roster.
        """
        async with self._pool.acquire() as conn:
            added = []
            dropped = []
            
            # Drop players first
            if drop_players:
                for name in drop_players:
                    result = await conn.execute(
                        "DELETE FROM rosters WHERE user_id = $1 AND player_name = $2",
                        user_id, name
                    )
                    if "DELETE 1" in result:
                        dropped.append(name)
            
            # Add players
            if add_players:
                for name in add_players:
                    await conn.execute(
                        """
                        INSERT INTO rosters (user_id, player_name, updated_at)
                        VALUES ($1, $2, NOW())
                        ON CONFLICT (user_id, player_name) DO NOTHING
                        """,
                        user_id, name
                    )
                    added.append(name)
            
            return {
                "added": added,
                "dropped": dropped,
                "add_count": len(added),
                "drop_count": len(dropped)
            }

    async def store_league_settings(
        self,
        user_id: str,
        league_name: str,
        scoring_format: str = "ppr",
        num_teams: int = 12,
        roster_slots: Optional[dict] = None
    ) -> dict:
        """Store or update league configuration."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO league_settings 
                    (user_id, league_name, scoring_format, num_teams, roster_slots, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (user_id) DO UPDATE
                SET league_name = $2, scoring_format = $3, num_teams = $4,
                    roster_slots = $5, updated_at = NOW()
                """,
                user_id, league_name, scoring_format, num_teams,
                json.dumps(roster_slots or {})
            )
            
            return {
                "user_id": user_id,
                "league_name": league_name,
                "scoring_format": scoring_format,
                "num_teams": num_teams,
                "roster_slots": roster_slots
            }

    # ─────────────────────────────────────────────────────────
    # Semantic Memory Operations (pgvector-powered)
    # ─────────────────────────────────────────────────────────

    async def log_interaction(
        self,
        user_id: str,
        summary: str,
        players_mentioned: Optional[List[str]] = None
    ) -> dict:
        """
        Store a conversation interaction as a searchable memory.
        
        THIS IS THE CORE OF SEMANTIC MEMORY.
        
        How it works:
          1. The summary text is converted to a 384-dim embedding vector
          2. The embedding + summary are stored in conversation_memory
          3. Later, search_past_advice() converts a query to an embedding
             and finds the most similar stored summaries using cosine similarity
        
        The agent should call this after giving significant advice,
        like a start/sit recommendation or trade analysis. Not every
        message needs to be logged — just the ones worth remembering.
        
        Args:
            user_id: Who this memory belongs to
            summary: Brief description of the advice given
            players_mentioned: Players discussed (for filtering)
        
        Returns:
            Confirmation dict with the memory ID
        """
        # Step 1: Generate embedding from the summary text
        embedding = self._embedder.generate(summary)
        
        # Step 2: Store in PostgreSQL with the pgvector embedding
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_memory 
                    (user_id, summary, players_mentioned, embedding, created_at)
                VALUES ($1, $2, $3, $4, NOW())
                RETURNING id, created_at
                """,
                user_id,
                summary,
                players_mentioned or [],
                json.dumps(embedding)  # pgvector accepts JSON array format
            )
            
            return {
                "memory_id": row["id"],
                "created_at": str(row["created_at"]),
                "summary_stored": summary[:100] + "..." if len(summary) > 100 else summary
            }

    async def search_past_advice(
        self,
        user_id: str,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[dict]:
        """
        Semantic search over a user's conversation history.
        
        THIS IS WHERE PGVECTOR SHINES.
        
        How it works:
          1. The search query is converted to an embedding vector
          2. pgvector computes cosine similarity between the query vector
             and ALL stored conversation embeddings for this user
          3. The most similar conversations are returned, ranked by relevance
        
        The SQL uses the <=> operator, which is pgvector's cosine distance.
        Lower distance = higher similarity (that's why we ORDER BY ASC).
        We convert to similarity score: 1 - distance = similarity (0 to 1).
        
        Example:
          Query: "what did you tell me about my running backs?"
          This gets embedded, then cosine-compared against stored memories like:
            - "Recommended Barkley over Henry due to matchup" (similarity: 0.85)
            - "Analyzed trade: giving Chase for Adams" (similarity: 0.21)
          The RB-related memory scores much higher and gets returned first.
        
        Args:
            user_id: Whose memories to search
            query: Natural language search query
            limit: Max results to return
            
        Returns:
            List of matching memories, sorted by relevance (most similar first)
        """
        # Step 1: Convert query to embedding
        query_embedding = self._embedder.generate(query)
        
        # Step 2: Perform cosine similarity search in pgvector
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    id,
                    summary,
                    players_mentioned,
                    created_at,
                    1 - (embedding <=> $1::vector) as similarity_score
                FROM conversation_memory
                WHERE user_id = $2
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $3
                """,
                json.dumps(query_embedding),
                user_id,
                limit
            )
            
            return [
                {
                    "memory_id": row["id"],
                    "summary": row["summary"],
                    "players_mentioned": row["players_mentioned"],
                    "created_at": str(row["created_at"]),
                    "relevance_score": round(float(row["similarity_score"]), 4)
                }
                for row in rows
            ]

    async def get_memory_count(self, user_id: str) -> int:
        """Get the total number of stored memories for a user."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM conversation_memory WHERE user_id = $1",
                user_id
            )
            return result or 0

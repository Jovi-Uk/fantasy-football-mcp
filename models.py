"""
Pydantic models for the Fantasy Football MCP Server.

These models define the input schemas for every tool the MCP server exposes.
FastMCP uses these models to:
  1. Auto-generate the JSON schema that clients see when they call list_tools()
  2. Validate all incoming parameters before your tool code ever runs
  3. Provide clear error messages when inputs are malformed

Think of these as contracts between the LLM agent and your server.
The agent sees the Field descriptions and knows exactly what to send.

Architecture note: We keep all models in one file so they're easy to find
and so multiple tools can share the same input types (DRY principle).
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ─────────────────────────────────────────────────────────────
# Shared Enums — Used across multiple tools
# ─────────────────────────────────────────────────────────────

class ScoringFormat(str, Enum):
    """Fantasy league scoring formats.
    
    The scoring format dramatically affects player values:
    - PPR (Point Per Reception): Favors pass-catching RBs and slot WRs
    - HALF_PPR: Balanced between rushing and receiving
    - STANDARD: Favors volume rushers and big-play receivers
    """
    PPR = "ppr"
    HALF_PPR = "half_ppr"
    STANDARD = "standard"


class Position(str, Enum):
    """NFL positions relevant to fantasy football."""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DEF = "DEF"


class ResponseFormat(str, Enum):
    """Output format for tool responses.
    
    MCP best practices say to support both:
    - MARKDOWN: Human-readable, great for display in chat UIs
    - JSON: Machine-readable, useful when the agent needs to process data further
    """
    MARKDOWN = "markdown"
    JSON = "json"


# ─────────────────────────────────────────────────────────────
# Data Tool Inputs — Fetching player/team information
# ─────────────────────────────────────────────────────────────

class PlayerStatsInput(BaseModel):
    """Input for retrieving player statistics."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    player_name: str = Field(
        ...,
        description="Full or partial player name (e.g., 'Derrick Henry', 'D. Henry', 'Henry')",
        min_length=2,
        max_length=100
    )
    season: Optional[int] = Field(
        default=2025,
        description="NFL season year (e.g., 2024, 2025)",
        ge=2020,
        le=2026
    )
    week: Optional[int] = Field(
        default=None,
        description="Specific week number (1-18). If None, returns season totals.",
        ge=1,
        le=18
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for structured data"
    )

    @field_validator('player_name')
    @classmethod
    def validate_player_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Player name cannot be empty")
        return v.strip()


class InjuryReportInput(BaseModel):
    """Input for retrieving injury reports."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    team: Optional[str] = Field(
        default=None,
        description="NFL team abbreviation (e.g., 'TEN', 'NYG', 'KC'). If None, returns league-wide injuries.",
        max_length=5
    )
    player_name: Optional[str] = Field(
        default=None,
        description="Filter by specific player name",
        max_length=100
    )
    position: Optional[Position] = Field(
        default=None,
        description="Filter by position (QB, RB, WR, TE, K, DEF)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class ComparePlayersInput(BaseModel):
    """Input for head-to-head player comparison."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    player_a: str = Field(
        ...,
        description="First player's name (e.g., 'Derrick Henry')",
        min_length=2,
        max_length=100
    )
    player_b: str = Field(
        ...,
        description="Second player's name (e.g., 'Saquon Barkley')",
        min_length=2,
        max_length=100
    )
    season: Optional[int] = Field(default=2025, description="NFL season year", ge=2020, le=2026)
    week: Optional[int] = Field(
        default=None,
        description="Compare for a specific week. If None, compares season averages.",
        ge=1, le=18
    )
    scoring_format: ScoringFormat = Field(
        default=ScoringFormat.PPR,
        description="League scoring format — affects fantasy point calculations"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ScheduleInput(BaseModel):
    """Input for retrieving team schedule and matchup difficulty."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    team: str = Field(
        ...,
        description="NFL team abbreviation (e.g., 'KC', 'SF', 'DAL')",
        min_length=2,
        max_length=5
    )
    weeks_ahead: Optional[int] = Field(
        default=4,
        description="Number of weeks to look ahead",
        ge=1, le=18
    )
    position: Optional[Position] = Field(
        default=None,
        description="Position to evaluate matchup difficulty against (e.g., 'RB' for run defense ranking)"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SearchPlayersInput(BaseModel):
    """Input for searching/filtering NFL players."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query: Optional[str] = Field(
        default=None,
        description="Search string to match against player names",
        max_length=100
    )
    position: Optional[Position] = Field(
        default=None,
        description="Filter by position"
    )
    team: Optional[str] = Field(
        default=None,
        description="Filter by team abbreviation",
        max_length=5
    )
    limit: Optional[int] = Field(
        default=20,
        description="Maximum number of results to return",
        ge=1, le=100
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class WaiverTargetsInput(BaseModel):
    """Input for finding waiver wire pickup targets."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    position: Position = Field(
        ...,
        description="Position to find waiver targets for (QB, RB, WR, TE)"
    )
    league_size: Optional[int] = Field(
        default=12,
        description="Number of teams in the league (affects who's likely available)",
        ge=8, le=20
    )
    scoring_format: ScoringFormat = Field(
        default=ScoringFormat.PPR,
        description="League scoring format"
    )
    limit: Optional[int] = Field(default=10, ge=1, le=25)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class PlayerTrendsInput(BaseModel):
    """Input for analyzing recent player performance trends."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    player_name: str = Field(..., description="Player name to analyze", min_length=2, max_length=100)
    num_weeks: Optional[int] = Field(
        default=4,
        description="Number of recent weeks to analyze",
        ge=2, le=10
    )
    season: Optional[int] = Field(default=2025, ge=2020, le=2026)
    scoring_format: ScoringFormat = Field(default=ScoringFormat.PPR)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


# ─────────────────────────────────────────────────────────────
# Analysis Tool Inputs — Higher-level reasoning tools
# ─────────────────────────────────────────────────────────────

class StartSitInput(BaseModel):
    """Input for start/sit analysis between two players."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    player_a: str = Field(..., description="Player you're considering starting", min_length=2, max_length=100)
    player_b: str = Field(..., description="Player you're considering sitting", min_length=2, max_length=100)
    week: Optional[int] = Field(default=None, description="Week to analyze (uses current week if None)", ge=1, le=18)
    scoring_format: ScoringFormat = Field(default=ScoringFormat.PPR)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class TradeEvalInput(BaseModel):
    """Input for evaluating a fantasy trade."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    give_players: List[str] = Field(
        ...,
        description="List of player names you would give away",
        min_length=1, max_length=5
    )
    receive_players: List[str] = Field(
        ...,
        description="List of player names you would receive",
        min_length=1, max_length=5
    )
    scoring_format: ScoringFormat = Field(default=ScoringFormat.PPR)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class OptimizeLineupInput(BaseModel):
    """Input for lineup optimization from a user's roster."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(..., description="User ID to pull roster from memory", min_length=1, max_length=100)
    week: Optional[int] = Field(default=None, description="Week to optimize for", ge=1, le=18)
    scoring_format: ScoringFormat = Field(default=ScoringFormat.PPR)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


# ─────────────────────────────────────────────────────────────
# Memory Tool Inputs — Roster & conversation persistence
# ─────────────────────────────────────────────────────────────

class StoreRosterInput(BaseModel):
    """Input for storing a user's fantasy roster."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(
        ...,
        description="Unique identifier for the user (e.g., username or session ID)",
        min_length=1, max_length=100
    )
    players: List[str] = Field(
        ...,
        description="List of player names on the roster (e.g., ['Derrick Henry', 'Ja\\'Marr Chase'])",
        min_length=1, max_length=25
    )
    league_name: Optional[str] = Field(
        default=None,
        description="Name of the fantasy league",
        max_length=100
    )
    scoring_format: ScoringFormat = Field(
        default=ScoringFormat.PPR,
        description="League scoring format"
    )


class GetRosterInput(BaseModel):
    """Input for retrieving a user's stored roster."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(..., description="User ID to look up", min_length=1, max_length=100)


class UpdateRosterInput(BaseModel):
    """Input for updating a user's roster (add/drop players)."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(..., description="User ID", min_length=1, max_length=100)
    add_players: Optional[List[str]] = Field(
        default_factory=list,
        description="Players to add to the roster",
        max_length=5
    )
    drop_players: Optional[List[str]] = Field(
        default_factory=list,
        description="Players to drop from the roster",
        max_length=5
    )


class StoreLeagueSettingsInput(BaseModel):
    """Input for storing league configuration."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(..., description="User ID", min_length=1, max_length=100)
    league_name: str = Field(..., description="League name", min_length=1, max_length=100)
    scoring_format: ScoringFormat = Field(default=ScoringFormat.PPR)
    num_teams: Optional[int] = Field(default=12, ge=4, le=20)
    roster_slots: Optional[dict] = Field(
        default=None,
        description="Roster slot configuration (e.g., {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1})"
    )


class SearchPastAdviceInput(BaseModel):
    """Input for semantic search over conversation history.
    
    This is where pgvector shines — the query gets converted to an embedding
    and matched against stored conversation embeddings using cosine similarity.
    """
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(..., description="User ID", min_length=1, max_length=100)
    query: str = Field(
        ...,
        description="Natural language query to search past advice (e.g., 'what did you say about my running backs')",
        min_length=3, max_length=500
    )
    limit: Optional[int] = Field(default=5, description="Max number of past interactions to retrieve", ge=1, le=20)


class LogInteractionInput(BaseModel):
    """Input for storing a conversation interaction for future retrieval.
    
    After each meaningful exchange, the agent can call this tool to store
    a summary of what was discussed. This builds the semantic memory layer.
    """
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(..., description="User ID", min_length=1, max_length=100)
    summary: str = Field(
        ...,
        description="Brief summary of the advice/interaction (e.g., 'Recommended starting Barkley over Henry due to matchup advantage')",
        min_length=10, max_length=1000
    )
    players_mentioned: Optional[List[str]] = Field(
        default_factory=list,
        description="Players discussed in this interaction"
    )

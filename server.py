#!/usr/bin/env python3
"""
Fantasy Football MCP Server — fantasy_football_mcp

This is the main entry point for the MCP server that powers the fantasy
football chatbot. It exposes tools, resources, and prompts that any
MCP-compatible client (like Dify, Claude Desktop, or a custom agent) can use.

=== HOW MCP WORKS (for the team) ===

When Dify (or any MCP client) connects to this server, here's what happens:

1. INITIALIZATION (Transport Layer → Lifecycle Management)
   Client sends an "initialize" request over the transport (HTTP or stdio).
   Server responds with its capabilities: what tools, resources, and prompts
   it offers. This is the "handshake" — the client now knows what it can ask for.

2. TOOL CALLS (Data Layer → Context Exchange)
   When the LLM agent decides it needs data (e.g., "get me Derrick Henry's stats"),
   it sends a "tools/call" request with the tool name and parameters.
   The server runs the tool function, fetches the data, and returns the result.
   The LLM then uses this data to formulate its response to the user.

3. RESOURCE ACCESS
   Resources are like RESTful endpoints — the client can browse them by URI.
   Example: "players://derrick-henry/profile" returns Henry's profile.
   Resources are for simple data access; tools are for complex operations.

4. PROMPT TEMPLATES
   The server provides pre-built prompt templates for common analysis tasks.
   The client can fetch these and use them as structured starting points.

=== ADVANCED MCP FEATURES USED ===

- Context Injection (ctx: Context): Every tool receives a Context object
  that provides logging, progress reporting, and access to shared state.

- Lifespan Management: Database connections and the player cache are
  initialized once at startup and shared across all tool calls.

- Tool Annotations: Each tool declares whether it's read-only, destructive,
  idempotent, etc. This helps the LLM agent make better decisions.

- Progress Reporting: Long-running tools report progress back to the client
  so the user knows something is happening.

=== RUNNING THE SERVER ===

Local development (STDIO transport — your current setup):
    python server.py

Production (Streamable HTTP transport — for team sharing):
    python server.py --transport http --port 8000

Then teammates connect Dify to: http://your-ip:8000/mcp
"""

import os
import json
import asyncio
import logging
import argparse
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.server import TransportSecuritySettings

# Import our modules
from models import (
    # Data tool inputs
    PlayerStatsInput, InjuryReportInput, ComparePlayersInput,
    ScheduleInput, SearchPlayersInput, WaiverTargetsInput,
    PlayerTrendsInput,
    # Analysis tool inputs
    StartSitInput, TradeEvalInput, OptimizeLineupInput,
    # Memory tool inputs
    StoreRosterInput, GetRosterInput, UpdateRosterInput,
    StoreLeagueSettingsInput, SearchPastAdviceInput, LogInteractionInput,
    # Enums
    ResponseFormat, ScoringFormat, Position
)
from data_providers import NFLDataProvider
from memory import MemoryManager

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Load from environment variables (see .env.example)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:fantasyfootball@localhost:5432/fantasy_bot"
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("fantasy_football_mcp")


# ─────────────────────────────────────────────────────────────
# Lifespan Management
# ─────────────────────────────────────────────────────────────
# 
# This is one of the most important MCP concepts to understand.
# 
# The lifespan function runs ONCE when the server starts up.
# Everything in the "yield" block stays alive for the server's lifetime.
# When the server shuts down, cleanup happens after the yield.
#
# Why this matters: Database connections, cached data, and the embedding
# model are expensive to initialize. We do it once here and share them
# across ALL tool calls via the Context system.
#
# In your tool function, you access these via:
#   ctx.request_context.lifespan_context["data_provider"]
#   ctx.request_context.lifespan_context["memory"]
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def server_lifespan(app):
    """
    Initialize shared resources that persist for the server's lifetime.
    
    This runs once at startup. All tools access these resources through
    the Context parameter — no global variables needed.
    """
    logger.info("🏈 Fantasy Football MCP Server starting up...")
    
    # Initialize the NFL data provider (loads player database from Sleeper)
    data_provider = NFLDataProvider()
    await data_provider.initialize()
    
    # Save player data to local cache for offline fallback
    await data_provider.save_cache()
    
    # Initialize the memory manager (connects to PostgreSQL + pgvector)
    memory = None
    try:
        memory = MemoryManager(DATABASE_URL)
        await memory.initialize()
        logger.info("✅ Memory manager (pgvector) connected and ready")
    except Exception as e:
        logger.warning(
            f"⚠️ Memory manager failed to initialize: {e}. "
            "Memory features will be unavailable. "
            "Make sure PostgreSQL + pgvector is running (see docker-compose.yml)"
        )
    
    logger.info("🏈 Server ready! Tools, resources, and prompts are available.")
    
    # Yield the shared state — tools access this via Context
    yield {
        "data_provider": data_provider,
        "memory": memory
    }
    
    # Cleanup on shutdown
    if memory:
        await memory.close()
    logger.info("🏈 Fantasy Football MCP Server shut down cleanly.")


# ─────────────────────────────────────────────────────────────
# Server Initialization
# ─────────────────────────────────────────────────────────────

mcp = FastMCP(
    "fantasy_football_mcp",
    lifespan=server_lifespan,
    # json_response=True enables structured JSON responses
    # which some clients prefer over text
    json_response=True,
    # Disable DNS rebinding protection to allow proxied traffic
    # (ngrok, Railway, Render, etc. forward requests with different Host headers)
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
)


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def _get_data_provider(ctx: Context) -> NFLDataProvider:
    """Extract the data provider from the lifespan context."""
    return ctx.request_context.lifespan_context["data_provider"]


def _get_memory(ctx: Context) -> Optional[MemoryManager]:
    """Extract the memory manager from the lifespan context."""
    return ctx.request_context.lifespan_context.get("memory")


def _require_memory(ctx: Context) -> MemoryManager:
    """Get memory manager or raise a helpful error if unavailable."""
    memory = _get_memory(ctx)
    if memory is None:
        raise RuntimeError(
            "Memory system is not available. Please ensure PostgreSQL + pgvector "
            "is running. Start it with: docker compose up -d"
        )
    return memory


# ═════════════════════════════════════════════════════════════
# DATA TOOLS — Fetching player and team information
# ═════════════════════════════════════════════════════════════


@mcp.tool(
    name="ff_get_player_stats",
    annotations={
        "title": "Get Player Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_get_player_stats(params: PlayerStatsInput, ctx: Context) -> str:
    """Retrieve detailed statistics for an NFL player.

    Fetches weekly or season-total stats from the Sleeper API, including
    passing, rushing, and receiving stats with fantasy point calculations.
    Use this when users ask about a player's performance.

    Args:
        params (PlayerStatsInput): Validated input containing:
            - player_name (str): Full or partial name (e.g., 'Derrick Henry')
            - season (Optional[int]): NFL season year (default: 2025)
            - week (Optional[int]): Specific week 1-18, or None for season totals
            - response_format: 'markdown' or 'json'

    Returns:
        str: Player stats in requested format, or error message if not found.
    """
    await ctx.report_progress(0.1, "Looking up player...")
    provider = _get_data_provider(ctx)
    
    result = await provider.get_player_stats(
        params.player_name, params.season, params.week
    )
    
    if not result:
        return (
            f"Player '{params.player_name}' not found. "
            "Try using the full name (e.g., 'Derrick Henry') or "
            "use ff_search_players to find the correct name."
        )
    
    await ctx.report_progress(0.8, "Formatting stats...")
    
    stats = result.get("stats", {})
    scoring = "ppr"  # Default; will be customizable
    fantasy_pts = provider.calculate_fantasy_points(stats, scoring)
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "player": provider.format_player_summary(result),
            "season": result.get("season"),
            "week": result.get("week"),
            "stats": stats,
            "fantasy_points_ppr": fantasy_pts
        }, indent=2)
    
    # Markdown format
    lines = [
        provider.format_player_summary(result),
        "",
        f"### {'Week ' + str(params.week) if params.week else 'Season'} Stats ({params.season})",
        provider.format_stats_line(stats),
        f"\n**Fantasy Points (PPR):** {fantasy_pts}",
    ]
    
    if result.get("games_played"):
        avg_pts = round(fantasy_pts / max(result["games_played"], 1), 2)
        lines.append(f"**Games Played:** {result['games_played']} | **Avg PPG:** {avg_pts}")
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_get_injury_report",
    annotations={
        "title": "Get Injury Report",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_get_injury_report(params: InjuryReportInput, ctx: Context) -> str:
    """Get current NFL injury report with status and body part details.

    Returns all injured players, optionally filtered by team, position,
    or player name. Includes injury designation (Out, Questionable, IR, etc.)
    and the injured body part.

    Args:
        params (InjuryReportInput): Validated input containing:
            - team (Optional[str]): Filter by team abbreviation
            - player_name (Optional[str]): Filter by player name
            - position (Optional[Position]): Filter by position
            - response_format: 'markdown' or 'json'

    Returns:
        str: Injury report in requested format.
    """
    provider = _get_data_provider(ctx)
    
    injuries = provider.get_injured_players(
        team=params.team,
        position=params.position.value if params.position else None,
        player_name=params.player_name
    )
    
    if not injuries:
        filter_desc = []
        if params.team:
            filter_desc.append(f"team={params.team}")
        if params.player_name:
            filter_desc.append(f"player={params.player_name}")
        filter_str = ", ".join(filter_desc) if filter_desc else "league-wide"
        return f"No injuries found ({filter_str}). All relevant players appear healthy."
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"injuries": injuries, "count": len(injuries)}, indent=2)
    
    # Markdown format
    lines = [f"## Injury Report ({len(injuries)} players)"]
    if params.team:
        lines[0] += f" — {params.team}"
    lines.append("")
    
    # Group by status severity
    severity_order = ["Out", "IR", "Doubtful", "Questionable", "PUP", "Sus"]
    for status in severity_order:
        group = [i for i in injuries if i["injury_status"] == status]
        if group:
            lines.append(f"### {status}")
            for player in group:
                lines.append(
                    f"- **{player['full_name']}** ({player['position']} — {player['team']}) "
                    f"— {player['injury_body_part']}"
                )
            lines.append("")
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_compare_players",
    annotations={
        "title": "Compare Two Players Head-to-Head",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_compare_players(params: ComparePlayersInput, ctx: Context) -> str:
    """Compare two players head-to-head with stats and fantasy point projections.

    Provides a detailed comparison including stats, fantasy points in the
    specified scoring format, injury status, and matchup context.
    Essential for start/sit decisions and trade evaluations.

    Args:
        params (ComparePlayersInput): Validated input containing:
            - player_a (str): First player's name
            - player_b (str): Second player's name
            - season (Optional[int]): Season year
            - week (Optional[int]): Week for comparison, or None for season avg
            - scoring_format (ScoringFormat): PPR, half_ppr, or standard
            - response_format: 'markdown' or 'json'

    Returns:
        str: Head-to-head comparison in requested format.
    """
    await ctx.report_progress(0.1, f"Looking up {params.player_a}...")
    provider = _get_data_provider(ctx)
    
    # Fetch both players' data
    player_a = await provider.get_player_stats(params.player_a, params.season, params.week)
    await ctx.report_progress(0.4, f"Looking up {params.player_b}...")
    player_b = await provider.get_player_stats(params.player_b, params.season, params.week)
    
    errors = []
    if not player_a:
        errors.append(f"Could not find '{params.player_a}'")
    if not player_b:
        errors.append(f"Could not find '{params.player_b}'")
    if errors:
        return " | ".join(errors) + ". Try using full player names."
    
    await ctx.report_progress(0.7, "Calculating comparison...")
    
    # Calculate fantasy points for both
    scoring = params.scoring_format.value
    pts_a = provider.calculate_fantasy_points(player_a.get("stats", {}), scoring)
    pts_b = provider.calculate_fantasy_points(player_b.get("stats", {}), scoring)
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "player_a": {
                "name": player_a.get("full_name"),
                "position": player_a.get("position"),
                "team": player_a.get("team"),
                "stats": player_a.get("stats", {}),
                "fantasy_points": pts_a,
                "injury_status": player_a.get("injury_status")
            },
            "player_b": {
                "name": player_b.get("full_name"),
                "position": player_b.get("position"),
                "team": player_b.get("team"),
                "stats": player_b.get("stats", {}),
                "fantasy_points": pts_b,
                "injury_status": player_b.get("injury_status")
            },
            "scoring_format": scoring,
            "advantage": player_a.get("full_name") if pts_a > pts_b else player_b.get("full_name"),
            "point_difference": round(abs(pts_a - pts_b), 2)
        }, indent=2)
    
    # Markdown comparison
    name_a = player_a.get("full_name", params.player_a)
    name_b = player_b.get("full_name", params.player_b)
    
    lines = [
        f"## Head-to-Head: {name_a} vs {name_b}",
        f"*Scoring: {scoring.upper()} | "
        f"{'Week ' + str(params.week) if params.week else 'Season ' + str(params.season)}*",
        "",
        f"### {name_a} ({player_a.get('position')} — {player_a.get('team')})",
        provider.format_stats_line(player_a.get("stats", {})),
        f"**Fantasy Points:** {pts_a}",
    ]
    
    if player_a.get("injury_status"):
        lines.append(f"⚠️ {player_a['injury_status']} ({player_a.get('injury_body_part', 'undisclosed')})")
    
    lines.extend([
        "",
        f"### {name_b} ({player_b.get('position')} — {player_b.get('team')})",
        provider.format_stats_line(player_b.get("stats", {})),
        f"**Fantasy Points:** {pts_b}",
    ])
    
    if player_b.get("injury_status"):
        lines.append(f"⚠️ {player_b['injury_status']} ({player_b.get('injury_body_part', 'undisclosed')})")
    
    # Verdict
    diff = round(abs(pts_a - pts_b), 2)
    winner = name_a if pts_a > pts_b else name_b
    lines.extend([
        "",
        f"### Verdict",
        f"**{winner}** has the edge by {diff} fantasy points in {scoring.upper()} scoring."
    ])
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_search_players",
    annotations={
        "title": "Search NFL Players",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def ff_search_players(params: SearchPlayersInput, ctx: Context) -> str:
    """Search and filter NFL players by name, position, or team.

    Useful for finding player names, exploring available players at a position,
    or checking who's on a specific team. Returns up to 'limit' results.

    Args:
        params (SearchPlayersInput): Validated input containing:
            - query (Optional[str]): Name search string
            - position (Optional[Position]): Filter by position
            - team (Optional[str]): Filter by team abbreviation
            - limit (int): Max results (default 20)
            - response_format: 'markdown' or 'json'

    Returns:
        str: List of matching players.
    """
    provider = _get_data_provider(ctx)
    
    results = provider.search_players(
        query=params.query,
        position=params.position.value if params.position else None,
        team=params.team,
        limit=params.limit
    )
    
    if not results:
        return f"No players found matching your criteria."
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "count": len(results),
            "players": [
                {
                    "name": p.get("full_name"),
                    "position": p.get("position"),
                    "team": p.get("team"),
                    "age": p.get("age"),
                    "injury_status": p.get("injury_status")
                }
                for p in results
            ]
        }, indent=2)
    
    lines = [f"## Player Search Results ({len(results)} found)"]
    for p in results:
        injury = f" ⚠️ {p.get('injury_status')}" if p.get("injury_status") else ""
        lines.append(
            f"- **{p.get('full_name')}** ({p.get('position')} — {p.get('team', 'FA')})"
            f" Age: {p.get('age', '?')}{injury}"
        )
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_get_waiver_targets",
    annotations={
        "title": "Find Waiver Wire Targets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_get_waiver_targets(params: WaiverTargetsInput, ctx: Context) -> str:
    """Suggest the best available waiver wire pickups at a position.

    Analyzes player stats and availability to recommend pickups that
    could improve a team. Factors in league size to estimate which
    players are likely available.

    Args:
        params (WaiverTargetsInput): Validated input containing:
            - position (Position): Position to target
            - league_size (int): Number of teams (affects availability)
            - scoring_format (ScoringFormat): League scoring format
            - limit (int): Max suggestions
            - response_format: 'markdown' or 'json'

    Returns:
        str: Ranked waiver wire suggestions with reasoning.
    """
    await ctx.report_progress(0.2, f"Scanning {params.position.value} waiver wire...")
    provider = _get_data_provider(ctx)
    
    # Get all active players at the position
    all_players = provider.search_players(
        position=params.position.value,
        limit=100
    )
    
    # In a real implementation, you'd cross-reference with roster ownership %
    # For now, we sort by some heuristic (years_exp, recent production)
    # and take the lower-ranked players who are likely available
    
    # Estimate the "waiver threshold" — players ranked beyond this
    # are likely available in a league of this size
    threshold = params.league_size * 2  # Rough heuristic
    
    # Sort by experience (proxy for name recognition / likely rostered)
    all_players.sort(key=lambda p: p.get("years_exp", 0))
    
    # Take players likely to be on waivers (less experienced, active)
    waiver_candidates = [
        p for p in all_players
        if p.get("team") is not None  # Must be on a team
    ][:params.limit * 2]  # Get extra to filter
    
    # Return the top suggestions
    targets = waiver_candidates[:params.limit]
    
    if not targets:
        return f"No waiver targets found at {params.position.value}."
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "position": params.position.value,
            "league_size": params.league_size,
            "targets": [
                {
                    "name": p.get("full_name"),
                    "team": p.get("team"),
                    "age": p.get("age"),
                    "experience": p.get("years_exp"),
                    "injury_status": p.get("injury_status")
                }
                for p in targets
            ]
        }, indent=2)
    
    lines = [
        f"## Waiver Wire: {params.position.value}",
        f"*{params.league_size}-team {params.scoring_format.value.upper()} league*",
        ""
    ]
    for i, p in enumerate(targets, 1):
        injury = f" ⚠️ {p.get('injury_status')}" if p.get("injury_status") else ""
        lines.append(
            f"{i}. **{p.get('full_name')}** ({p.get('team', 'FA')})"
            f" — {p.get('years_exp', '?')} yr exp{injury}"
        )
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_get_player_trends",
    annotations={
        "title": "Analyze Player Performance Trends",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_get_player_trends(params: PlayerTrendsInput, ctx: Context) -> str:
    """Analyze a player's recent performance trajectory over multiple weeks.

    Shows week-by-week fantasy point production to identify trends like
    hot streaks, cold spells, increasing targets, or declining usage.
    Crucial for buy-low/sell-high decisions.

    Args:
        params (PlayerTrendsInput): Validated input containing:
            - player_name (str): Player to analyze
            - num_weeks (int): Number of recent weeks to include
            - season (int): Season year
            - scoring_format (ScoringFormat): For point calculations
            - response_format: 'markdown' or 'json'

    Returns:
        str: Week-by-week trend analysis.
    """
    await ctx.report_progress(0.1, f"Analyzing trends for {params.player_name}...")
    provider = _get_data_provider(ctx)
    
    # Find the player first
    player = provider.find_player(params.player_name)
    if not player:
        return f"Player '{params.player_name}' not found."
    
    pid = player["player_id"]
    scoring = params.scoring_format.value
    
    # Collect weekly data
    weekly_data = []
    for week in range(1, params.num_weeks + 1):
        await ctx.report_progress(
            0.1 + (0.7 * week / params.num_weeks),
            f"Fetching week {week}..."
        )
        stats = await provider.get_weekly_stats(params.season, week)
        if pid in stats:
            pts = provider.calculate_fantasy_points(stats[pid], scoring)
            weekly_data.append({
                "week": week,
                "stats": stats[pid],
                "fantasy_points": pts
            })
    
    if not weekly_data:
        return f"No stats found for {player.get('full_name')} in the last {params.num_weeks} weeks."
    
    # Calculate trend metrics
    points = [w["fantasy_points"] for w in weekly_data]
    avg_pts = round(sum(points) / len(points), 2)
    max_pts = max(points)
    min_pts = min(points)
    
    # Simple trend direction: compare first half average to second half
    mid = len(points) // 2
    first_half_avg = sum(points[:mid]) / max(mid, 1)
    second_half_avg = sum(points[mid:]) / max(len(points) - mid, 1)
    trend = "📈 Trending UP" if second_half_avg > first_half_avg else "📉 Trending DOWN"
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "player": player.get("full_name"),
            "weeks": weekly_data,
            "average": avg_pts,
            "high": max_pts,
            "low": min_pts,
            "trend": trend
        }, indent=2)
    
    lines = [
        f"## {player.get('full_name')} — Performance Trends",
        f"*Last {len(weekly_data)} weeks | {scoring.upper()} scoring*",
        "",
        f"**Average:** {avg_pts} pts | **High:** {max_pts} | **Low:** {min_pts} | {trend}",
        "",
        "### Week-by-Week"
    ]
    
    for w in weekly_data:
        bar_length = int(w["fantasy_points"] / max_pts * 20) if max_pts > 0 else 0
        bar = "█" * bar_length
        lines.append(f"Wk {w['week']:2d}: {w['fantasy_points']:6.1f} pts {bar}")
    
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# ANALYSIS TOOLS — Higher-level reasoning
# ═════════════════════════════════════════════════════════════


@mcp.tool(
    name="ff_analyze_start_sit",
    annotations={
        "title": "Start/Sit Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_analyze_start_sit(params: StartSitInput, ctx: Context) -> str:
    """Provide a data-driven start/sit recommendation between two players.

    Analyzes stats, injury status, matchup, and recent trends to recommend
    which player to start and which to sit, with detailed reasoning.

    Args:
        params (StartSitInput): Validated input containing:
            - player_a (str): First player name
            - player_b (str): Second player name
            - week (Optional[int]): Week to analyze
            - scoring_format (ScoringFormat): League scoring
            - response_format: 'markdown' or 'json'

    Returns:
        str: Start/sit recommendation with reasoning.
    """
    await ctx.report_progress(0.1, "Analyzing matchups...")
    provider = _get_data_provider(ctx)
    scoring = params.scoring_format.value
    
    # Get both players' data
    a_data = await provider.get_player_stats(params.player_a, 2025, params.week)
    b_data = await provider.get_player_stats(params.player_b, 2025, params.week)
    
    if not a_data or not b_data:
        not_found = []
        if not a_data:
            not_found.append(params.player_a)
        if not b_data:
            not_found.append(params.player_b)
        return f"Could not find: {', '.join(not_found)}"
    
    await ctx.report_progress(0.6, "Computing recommendation...")
    
    # Calculate fantasy points
    pts_a = provider.calculate_fantasy_points(a_data.get("stats", {}), scoring)
    pts_b = provider.calculate_fantasy_points(b_data.get("stats", {}), scoring)
    
    # Factor in injury risk
    injury_a = a_data.get("injury_status")
    injury_b = b_data.get("injury_status")
    
    injury_penalty = {"Out": -100, "Doubtful": -10, "Questionable": -2, "IR": -100, "PUP": -100}
    adjusted_a = pts_a + injury_penalty.get(injury_a, 0)
    adjusted_b = pts_b + injury_penalty.get(injury_b, 0)
    
    # Determine recommendation
    start_player = a_data if adjusted_a >= adjusted_b else b_data
    sit_player = b_data if adjusted_a >= adjusted_b else a_data
    start_pts = max(adjusted_a, adjusted_b)
    sit_pts = min(adjusted_a, adjusted_b)
    
    name_a = a_data.get("full_name")
    name_b = b_data.get("full_name")
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "recommendation": "START",
            "start": start_player.get("full_name"),
            "sit": sit_player.get("full_name"),
            "start_projected_pts": round(start_pts, 2),
            "sit_projected_pts": round(sit_pts, 2),
            "confidence": "HIGH" if abs(start_pts - sit_pts) > 5 else "MODERATE" if abs(start_pts - sit_pts) > 2 else "LOW"
        }, indent=2)
    
    confidence = "HIGH" if abs(start_pts - sit_pts) > 5 else "MODERATE" if abs(start_pts - sit_pts) > 2 else "TOSS-UP"
    
    lines = [
        f"## Start/Sit: {name_a} vs {name_b}",
        f"*{scoring.upper()} scoring*",
        "",
        f"### ✅ START: {start_player.get('full_name')} ({round(start_pts, 1)} pts projected)",
        f"### ❌ SIT: {sit_player.get('full_name')} ({round(sit_pts, 1)} pts projected)",
        "",
        f"**Confidence:** {confidence}",
        "",
        "### Reasoning",
    ]
    
    reasons = []
    if pts_a != pts_b:
        reasons.append(
            f"Based on production, {name_a if pts_a > pts_b else name_b} "
            f"has outperformed by {abs(pts_a - pts_b):.1f} fantasy points."
        )
    if injury_a:
        reasons.append(f"⚠️ {name_a} is listed as {injury_a} ({a_data.get('injury_body_part', 'undisclosed')})")
    if injury_b:
        reasons.append(f"⚠️ {name_b} is listed as {injury_b} ({b_data.get('injury_body_part', 'undisclosed')})")
    
    if not reasons:
        reasons.append("Both players are healthy with similar production. Consider matchup difficulty.")
    
    for r in reasons:
        lines.append(f"- {r}")
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_evaluate_trade",
    annotations={
        "title": "Evaluate Fantasy Trade",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def ff_evaluate_trade(params: TradeEvalInput, ctx: Context) -> str:
    """Analyze a proposed fantasy trade and recommend whether to accept or reject.

    Compares the total fantasy value of players being given away vs received,
    accounting for position scarcity, injury risk, and scoring format.

    Args:
        params (TradeEvalInput): Validated input containing:
            - give_players (List[str]): Players you'd trade away
            - receive_players (List[str]): Players you'd receive
            - scoring_format (ScoringFormat): League scoring
            - response_format: 'markdown' or 'json'

    Returns:
        str: Trade evaluation with recommendation.
    """
    await ctx.report_progress(0.1, "Analyzing trade values...")
    provider = _get_data_provider(ctx)
    scoring = params.scoring_format.value
    
    # Calculate total value of each side
    give_value = 0.0
    give_details = []
    for name in params.give_players:
        data = await provider.get_player_stats(name, 2025)
        if data:
            pts = provider.calculate_fantasy_points(data.get("stats", {}), scoring)
            give_value += pts
            give_details.append({"name": data.get("full_name"), "pts": pts})
        else:
            give_details.append({"name": name, "pts": 0, "error": "not found"})
    
    await ctx.report_progress(0.5, "Calculating receive value...")
    
    receive_value = 0.0
    receive_details = []
    for name in params.receive_players:
        data = await provider.get_player_stats(name, 2025)
        if data:
            pts = provider.calculate_fantasy_points(data.get("stats", {}), scoring)
            receive_value += pts
            receive_details.append({"name": data.get("full_name"), "pts": pts})
        else:
            receive_details.append({"name": name, "pts": 0, "error": "not found"})
    
    diff = receive_value - give_value
    verdict = "ACCEPT" if diff > 0 else "REJECT" if diff < -5 else "FAIR TRADE"
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "verdict": verdict,
            "give_total": round(give_value, 2),
            "receive_total": round(receive_value, 2),
            "net_value": round(diff, 2),
            "give_details": give_details,
            "receive_details": receive_details
        }, indent=2)
    
    lines = [
        f"## Trade Evaluation",
        f"*{scoring.upper()} scoring*",
        "",
        f"### You Give ({round(give_value, 1)} total pts):"
    ]
    for d in give_details:
        lines.append(f"- {d['name']}: {d['pts']:.1f} pts")
    
    lines.extend([
        "",
        f"### You Receive ({round(receive_value, 1)} total pts):"
    ])
    for d in receive_details:
        lines.append(f"- {d['name']}: {d['pts']:.1f} pts")
    
    emoji = "✅" if verdict == "ACCEPT" else "❌" if verdict == "REJECT" else "🤝"
    lines.extend([
        "",
        f"### {emoji} Verdict: {verdict}",
        f"Net value change: {'+' if diff > 0 else ''}{diff:.1f} fantasy points"
    ])
    
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MEMORY TOOLS — Roster & conversation persistence
# ═════════════════════════════════════════════════════════════


@mcp.tool(
    name="ff_store_roster",
    annotations={
        "title": "Store User's Fantasy Roster",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def ff_store_roster(params: StoreRosterInput, ctx: Context) -> str:
    """Save a user's fantasy roster to persistent memory.

    Stores the roster in PostgreSQL so it persists across sessions.
    The agent should call this when a user shares their roster.
    Subsequent advice can then reference the stored roster.

    Args:
        params (StoreRosterInput): Validated input containing:
            - user_id (str): Unique user identifier
            - players (List[str]): Player names on the roster
            - league_name (Optional[str]): League name
            - scoring_format (ScoringFormat): League format

    Returns:
        str: Confirmation of stored roster.
    """
    memory = _require_memory(ctx)
    
    result = await memory.store_roster(
        user_id=params.user_id,
        players=params.players,
        scoring_format=params.scoring_format.value,
        league_name=params.league_name
    )
    
    return json.dumps({
        "status": "success",
        "message": f"Stored {result['stored_count']} players for user {params.user_id}",
        **result
    }, indent=2)


@mcp.tool(
    name="ff_get_roster",
    annotations={
        "title": "Get User's Stored Roster",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def ff_get_roster(params: GetRosterInput, ctx: Context) -> str:
    """Retrieve a user's stored fantasy roster from memory.

    Returns the saved roster and league settings. Use this before giving
    personalized advice to know what players the user has.

    Args:
        params (GetRosterInput): Validated input containing:
            - user_id (str): User ID to look up

    Returns:
        str: Roster details or message if no roster found.
    """
    memory = _require_memory(ctx)
    
    roster = await memory.get_roster(params.user_id)
    
    if not roster:
        return (
            f"No roster found for user '{params.user_id}'. "
            "Ask the user to share their roster, then use ff_store_roster to save it."
        )
    
    return json.dumps(roster, indent=2, default=str)


@mcp.tool(
    name="ff_update_roster",
    annotations={
        "title": "Update User's Roster (Add/Drop)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def ff_update_roster(params: UpdateRosterInput, ctx: Context) -> str:
    """Add or drop players from a user's stored roster.

    Used when a user makes a waiver claim or free agent pickup.
    More efficient than re-storing the entire roster.

    Args:
        params (UpdateRosterInput): Validated input containing:
            - user_id (str): User ID
            - add_players (Optional[List[str]]): Players to add
            - drop_players (Optional[List[str]]): Players to drop

    Returns:
        str: Summary of roster changes.
    """
    memory = _require_memory(ctx)
    
    result = await memory.update_roster(
        user_id=params.user_id,
        add_players=params.add_players,
        drop_players=params.drop_players
    )
    
    return json.dumps({
        "status": "success",
        "message": f"Added {result['add_count']}, dropped {result['drop_count']}",
        **result
    }, indent=2)


@mcp.tool(
    name="ff_store_league_settings",
    annotations={
        "title": "Store League Settings",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def ff_store_league_settings(params: StoreLeagueSettingsInput, ctx: Context) -> str:
    """Save a user's fantasy league configuration.

    Stores scoring format, league size, and roster slot rules so
    future advice is tailored to the user's specific league.

    Args:
        params (StoreLeagueSettingsInput): Validated input containing:
            - user_id (str): User ID
            - league_name (str): League name
            - scoring_format (ScoringFormat): Scoring format
            - num_teams (int): Number of teams
            - roster_slots (Optional[dict]): Slot configuration

    Returns:
        str: Confirmation of stored settings.
    """
    memory = _require_memory(ctx)
    
    result = await memory.store_league_settings(
        user_id=params.user_id,
        league_name=params.league_name,
        scoring_format=params.scoring_format.value,
        num_teams=params.num_teams,
        roster_slots=params.roster_slots
    )
    
    return json.dumps({"status": "success", **result}, indent=2)


@mcp.tool(
    name="ff_search_past_advice",
    annotations={
        "title": "Search Past Advice (Semantic Memory)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def ff_search_past_advice(params: SearchPastAdviceInput, ctx: Context) -> str:
    """Semantically search through past conversation history using pgvector.

    Uses vector similarity search to find the most relevant past advice
    given to this user. The query is embedded and compared against stored
    conversation embeddings using cosine similarity.

    This is the key feature that gives the chatbot genuine memory —
    it can recall what it said days or weeks ago.

    Args:
        params (SearchPastAdviceInput): Validated input containing:
            - user_id (str): User ID
            - query (str): Natural language search query
            - limit (int): Max results

    Returns:
        str: Most relevant past interactions, ranked by similarity.
    """
    memory = _require_memory(ctx)
    
    await ctx.report_progress(0.3, "Searching conversation history...")
    
    results = await memory.search_past_advice(
        user_id=params.user_id,
        query=params.query,
        limit=params.limit
    )
    
    if not results:
        return (
            f"No past conversations found for user '{params.user_id}'. "
            "Past advice is stored when ff_log_interaction is called after giving recommendations."
        )
    
    lines = [
        f"## Past Advice (Top {len(results)} matches)",
        f"*Query: \"{params.query}\"*",
        ""
    ]
    
    for r in results:
        relevance_pct = round(r["relevance_score"] * 100, 1)
        lines.append(f"### Match ({relevance_pct}% relevant)")
        lines.append(f"*{r['created_at']}*")
        lines.append(r["summary"])
        if r.get("players_mentioned"):
            lines.append(f"Players: {', '.join(r['players_mentioned'])}")
        lines.append("")
    
    return "\n".join(lines)


@mcp.tool(
    name="ff_log_interaction",
    annotations={
        "title": "Log Conversation for Future Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def ff_log_interaction(params: LogInteractionInput, ctx: Context) -> str:
    """Store a conversation summary as a searchable memory.

    Call this after giving meaningful advice to build the user's
    conversation history. The summary is embedded as a vector and
    stored in pgvector for future semantic search.

    Args:
        params (LogInteractionInput): Validated input containing:
            - user_id (str): User ID
            - summary (str): Brief description of the advice given
            - players_mentioned (Optional[List[str]]): Players discussed

    Returns:
        str: Confirmation with memory ID.
    """
    memory = _require_memory(ctx)
    
    result = await memory.log_interaction(
        user_id=params.user_id,
        summary=params.summary,
        players_mentioned=params.players_mentioned
    )
    
    return json.dumps({
        "status": "success",
        "message": "Interaction logged to semantic memory",
        **result
    }, indent=2)


# ═════════════════════════════════════════════════════════════
# MCP RESOURCES — Browsable data endpoints
# ═════════════════════════════════════════════════════════════
#
# Resources are like RESTful GET endpoints. The client can
# browse available resources and fetch data by URI template.
# They're simpler than tools — no complex parameters, just URIs.
# ═════════════════════════════════════════════════════════════


@mcp.resource("players://{player_name}/profile")
async def player_profile_resource(player_name: str) -> str:
    """
    Get a player's profile by name.
    
    Browsable via URI: players://derrick-henry/profile
    
    Note: Resources can't access Context directly in the same way tools can.
    For the resource layer, we create a temporary provider instance.
    In production, you'd use a shared singleton.
    """
    provider = NFLDataProvider()
    # Check if already initialized (from lifespan)
    if not provider._players:
        await provider.initialize()
    
    player = provider.find_player(player_name.replace("-", " "))
    if not player:
        return json.dumps({"error": f"Player '{player_name}' not found"})
    
    return json.dumps({
        "name": player.get("full_name"),
        "position": player.get("position"),
        "team": player.get("team"),
        "age": player.get("age"),
        "experience": player.get("years_exp"),
        "height": player.get("height"),
        "weight": player.get("weight"),
        "injury_status": player.get("injury_status"),
        "injury_body_part": player.get("injury_body_part"),
        "status": player.get("status")
    }, indent=2)


@mcp.resource("teams://list")
async def teams_list_resource() -> str:
    """List all NFL teams with abbreviations."""
    from data_providers import NFL_TEAMS
    return json.dumps({"teams": sorted(NFL_TEAMS)}, indent=2)


# ═════════════════════════════════════════════════════════════
# MCP PROMPTS — Pre-built prompt templates
# ═════════════════════════════════════════════════════════════
#
# Prompts are templates that the client can request and use
# as structured starting points for LLM interactions.
# Think of them as "recipes" for common analysis patterns.
# The client fetches the prompt, fills in the variables,
# and sends it to the LLM along with any tool results.
# ═════════════════════════════════════════════════════════════


@mcp.prompt()
def ff_start_sit_decision(player_a: str, player_b: str, week: str = "current") -> str:
    """Generate a structured start/sit analysis prompt.
    
    Use this prompt template when a user needs help deciding between
    two players for their weekly lineup.
    """
    return f"""You are an expert fantasy football analyst. A manager needs help 
deciding between two players for Week {week}.

**PLAYER A:** {player_a}
**PLAYER B:** {player_b}

Please analyze this decision by:
1. First, use ff_get_player_stats to pull each player's recent stats
2. Use ff_get_injury_report to check both players' health status
3. Use ff_compare_players for a head-to-head breakdown
4. Consider matchup difficulty using ff_get_player_trends

Provide your recommendation with:
- A clear START/SIT verdict
- 3 key reasons supporting your choice
- Risk factors to consider
- Confidence level (HIGH/MODERATE/LOW)"""


@mcp.prompt()
def ff_trade_analyzer(give: str, receive: str) -> str:
    """Generate a structured trade analysis prompt."""
    return f"""You are an expert fantasy football trade analyst. Evaluate this trade:

**GIVING:** {give}
**RECEIVING:** {receive}

Please analyze by:
1. Use ff_evaluate_trade to get the raw value comparison
2. Use ff_get_player_trends to check recent trajectories
3. Consider position scarcity and remaining schedule

Provide:
- ACCEPT/REJECT/COUNTER recommendation
- Value breakdown for each side
- Roster impact analysis
- Long-term vs short-term tradeoffs"""


@mcp.prompt()
def ff_weekly_review(user_id: str) -> str:
    """Generate a weekly roster review prompt."""
    return f"""You are a fantasy football advisor reviewing a manager's roster.

1. First, use ff_get_roster to pull the user's roster (user_id: {user_id})
2. For each player, check ff_get_injury_report for health updates
3. Identify any concerning trends using ff_get_player_trends
4. Check if any ff_get_waiver_targets could improve the roster

Provide:
- Overall roster grade (A through F)
- Position-by-position assessment
- Top 3 action items for the week
- Waiver wire recommendations if applicable"""


@mcp.prompt()
def ff_waiver_strategy(position: str, league_size: str = "12") -> str:
    """Generate a waiver wire strategy prompt."""
    return f"""You are a waiver wire specialist. Help find the best available 
{position} pickups for a {league_size}-team league.

1. Use ff_get_waiver_targets for the {position} position
2. For each suggestion, use ff_get_player_trends to verify upward trajectory  
3. Check ff_get_injury_report for any health concerns
4. Consider upcoming schedule difficulty

Rank your top 5 recommendations with:
- Why they're available (low ownership)
- Their upside case
- Their floor (worst case)
- Priority ranking for FAAB/waiver order"""


# ═════════════════════════════════════════════════════════════
# SERVER ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fantasy Football MCP Server")
    # If PORT env var is set (e.g., by Railway/Render), default to HTTP transport
    default_transport = "http" if os.getenv("PORT") else "stdio"
    
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=default_transport,
        help="Transport method: 'stdio' for local dev, 'http' for network access"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port for HTTP transport (default: 8000, or PORT env var)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    if args.transport == "http":
        logger.info(f"Starting with Streamable HTTP transport on {args.host}:{args.port}")
        # Host and port are set via FastMCP settings, not run() kwargs
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        logger.info("Starting with STDIO transport (local development)")
        mcp.run(transport="stdio")

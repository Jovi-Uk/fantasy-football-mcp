"""
Data Providers for the Fantasy Football MCP Server.

This module is the data backbone of the entire system. It handles fetching
NFL player data, stats, injuries, and projections from external sources.

Primary data source: Sleeper API (https://api.sleeper.app)
  - Free, no authentication required
  - Covers all NFL players, weekly stats, projections
  - Rate-limited but generous for our use case

Architecture decisions:
  1. We cache the full player database in memory on startup (it's ~15MB)
     because the Sleeper /players/nfl endpoint is slow and the data
     only changes once per day. This is loaded during server lifespan.
  
  2. Weekly stats are fetched on-demand but cached per (season, week) pair
     since historical stats never change once a week is complete.
  
  3. We use httpx for async HTTP because MCP tools are async and we don't
     want to block the event loop with synchronous requests.

If Sleeper is down or rate-limited, tools will return helpful error messages
rather than crashing — following MCP best practices for actionable errors.
"""

import json
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import httpx

logger = logging.getLogger("fantasy_football_mcp.data")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
REQUEST_TIMEOUT = 30.0

# NFL team abbreviations for validation
NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
}

# Fantasy-relevant positions
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


# ─────────────────────────────────────────────────────────────
# Core Data Provider Class
# ─────────────────────────────────────────────────────────────

class NFLDataProvider:
    """
    Handles all NFL data fetching, caching, and lookups.
    
    This class is instantiated once during server lifespan and shared
    across all tool invocations via the MCP Context system.
    
    The player database is a dictionary keyed by Sleeper player_id,
    containing metadata like name, position, team, injury status, etc.
    
    Usage:
        provider = NFLDataProvider()
        await provider.initialize()  # Called during server startup
        stats = await provider.get_player_stats("Derrick Henry", 2025, 10)
    """

    def __init__(self):
        # The full player database, loaded from Sleeper on startup
        # Key: player_id (str), Value: player dict with metadata
        self._players: Dict[str, dict] = {}
        
        # Name-to-ID lookup for fast player search
        # Key: lowercase full name, Value: player_id
        self._name_index: Dict[str, str] = {}
        
        # Cache for weekly stats: (season, week) -> {player_id: stats_dict}
        self._stats_cache: Dict[tuple, dict] = {}
        
        # Cache for projections
        self._projections_cache: Dict[tuple, dict] = {}

    async def initialize(self):
        """
        Load the player database from Sleeper API.
        
        This is called once during server startup (via lifespan).
        The Sleeper /players/nfl endpoint returns ALL NFL players (~10,000+)
        as a massive JSON object. We cache this in memory because:
          1. It only changes ~once per day
          2. It's needed by almost every tool
          3. The endpoint is slow (~2-5 seconds)
        """
        logger.info("Initializing NFL data provider — loading player database...")
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(f"{SLEEPER_BASE_URL}/players/nfl")
                response.raise_for_status()
                self._players = response.json()
            
            # Build the name index for fast lookups
            self._build_name_index()
            
            logger.info(
                f"Player database loaded: {len(self._players)} players, "
                f"{len(self._name_index)} indexed names"
            )
            
        except httpx.HTTPError as e:
            logger.warning(f"Failed to load from Sleeper API: {e}. Trying local cache...")
            await self._load_from_cache()

    def _build_name_index(self):
        """
        Build a lookup index from player names to player IDs.
        
        This enables fuzzy-ish matching: we index by full name, last name,
        and common abbreviations so the LLM agent doesn't need to send
        an exact match.
        
        IMPORTANT: For last-name and abbreviation keys (where collisions are
        likely — e.g., multiple players named "Henry"), we use Sleeper's
        'search_rank' field to prefer the most fantasy-relevant player.
        Lower search_rank = more relevant. So Derrick Henry (search_rank ~5)
        beats Zuri Henry (search_rank ~9999) for the "henry" key.
        
        Full name keys never collide (they're unique), so no ranking needed there.
        """
        self._name_index = {}
        
        # We also track the search_rank of whoever currently owns each key,
        # so we can replace them if a more relevant player comes along.
        # This only applies to last-name and abbreviation keys.
        self._index_ranks: dict[str, int] = {}
        
        for pid, player in self._players.items():
            if not isinstance(player, dict):
                continue
            
            full_name = player.get("full_name", "")
            if not full_name:
                continue
            
            # Sleeper's search_rank: lower = more fantasy-relevant.
            # Players without a rank get a very high number (least priority).
            player_rank = player.get("search_rank", 999999) or 999999
            
            # Always index by full name (these are unique, no collision logic needed)
            self._name_index[full_name.lower()] = pid
            
            # Index by last name — but prefer the highest-ranked player
            last_name = player.get("last_name", "")
            if last_name:
                key = last_name.lower()
                existing_rank = self._index_ranks.get(key, 999999)
                if key not in self._name_index or player_rank < existing_rank:
                    self._name_index[key] = pid
                    self._index_ranks[key] = player_rank
            
            # Index by "F. LastName" format (e.g., "D. Henry") — same ranking logic
            first_name = player.get("first_name", "")
            if first_name and last_name:
                abbrev = f"{first_name[0].lower()}. {last_name.lower()}"
                existing_rank = self._index_ranks.get(abbrev, 999999)
                if abbrev not in self._name_index or player_rank < existing_rank:
                    self._name_index[abbrev] = pid
                    self._index_ranks[abbrev] = player_rank

    async def _load_from_cache(self):
        """Fallback: load player data from local cache file if API is unavailable."""
        cache_file = CACHE_DIR / "players.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                self._players = json.load(f)
            self._build_name_index()
            logger.info(f"Loaded {len(self._players)} players from local cache")
        else:
            logger.error("No local cache available. Player data will be empty.")
            self._players = {}

    async def save_cache(self):
        """Save current player data to local cache for offline use."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / "players.json"
        with open(cache_file, "w") as f:
            json.dump(self._players, f)
        logger.info(f"Saved {len(self._players)} players to local cache")

    # ─────────────────────────────────────────────────────────
    # Player Lookup Methods
    # ─────────────────────────────────────────────────────────

    def find_player(self, name: str) -> Optional[dict]:
        """
        Find a player by name with fuzzy matching.
        
        Search strategy (in order of specificity):
          1. Exact full name match
          2. Exact last name match
          3. Abbreviation match (e.g., "D. Henry")
          4. Substring match (finds "Henry" in "Derrick Henry")
        
        Returns the player dict or None if not found.
        """
        name_lower = name.lower().strip()
        
        # Strategy 1: Exact match in index
        if name_lower in self._name_index:
            pid = self._name_index[name_lower]
            return {**self._players[pid], "player_id": pid}
        
        # Strategy 2: Substring match across all indexed names
        matches = []
        for indexed_name, pid in self._name_index.items():
            if name_lower in indexed_name or indexed_name in name_lower:
                player = self._players.get(pid)
                if player and isinstance(player, dict):
                    matches.append({**player, "player_id": pid})
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Prefer fantasy-relevant positions when multiple matches
            fantasy_matches = [
                m for m in matches 
                if m.get("position") in FANTASY_POSITIONS
                and m.get("status") == "Active"
            ]
            if fantasy_matches:
                return fantasy_matches[0]
            return matches[0]
        
        return None

    def search_players(
        self,
        query: Optional[str] = None,
        position: Optional[str] = None,
        team: Optional[str] = None,
        limit: int = 20
    ) -> List[dict]:
        """
        Search and filter players with multiple criteria.
        
        Used by the ff_search_players tool to let agents explore
        the player database with flexible queries.
        """
        results = []
        
        for pid, player in self._players.items():
            if not isinstance(player, dict):
                continue
            
            # Only include fantasy-relevant, active players
            if player.get("position") not in FANTASY_POSITIONS:
                continue
            if player.get("status") != "Active" and player.get("active") is not True:
                continue
            
            # Apply filters
            if position and player.get("position") != position:
                continue
            if team and player.get("team") != team:
                continue
            if query:
                full_name = player.get("full_name", "").lower()
                if query.lower() not in full_name:
                    continue
            
            results.append({**player, "player_id": pid})
            
            if len(results) >= limit:
                break
        
        return results

    def get_injured_players(
        self,
        team: Optional[str] = None,
        position: Optional[str] = None,
        player_name: Optional[str] = None
    ) -> List[dict]:
        """
        Get all players with an injury designation.
        
        Sleeper tracks injury_status as one of:
          - "Questionable", "Doubtful", "Out", "IR" (Injured Reserve)
          - "PUP" (Physically Unable to Perform)
          - "Sus" (Suspended)
          - None (healthy)
        
        The injury_body_part field tells you what's injured.
        """
        injured = []
        
        for pid, player in self._players.items():
            if not isinstance(player, dict):
                continue
            
            injury_status = player.get("injury_status")
            if not injury_status:
                continue
            
            # Only fantasy-relevant positions
            if player.get("position") not in FANTASY_POSITIONS:
                continue
            
            # Apply filters
            if team and player.get("team") != team:
                continue
            if position and player.get("position") != position:
                continue
            if player_name:
                full_name = player.get("full_name", "").lower()
                if player_name.lower() not in full_name:
                    continue
            
            injured.append({
                "player_id": pid,
                "full_name": player.get("full_name"),
                "position": player.get("position"),
                "team": player.get("team"),
                "injury_status": injury_status,
                "injury_body_part": player.get("injury_body_part", "Unknown"),
                "injury_start_date": player.get("injury_start_date"),
                "injury_notes": player.get("injury_notes", ""),
            })
        
        return injured

    # ─────────────────────────────────────────────────────────
    # Stats Fetching
    # ─────────────────────────────────────────────────────────

    async def get_weekly_stats(self, season: int, week: int) -> Dict[str, dict]:
        """
        Fetch weekly stats for all players from Sleeper.
        
        Returns a dict keyed by player_id with their stats for that week.
        Stats include rushing_yds, receiving_yds, passing_tds, etc.
        
        Results are cached because historical stats don't change.
        """
        cache_key = (season, week)
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{SLEEPER_BASE_URL}/stats/nfl/regular/{season}/{week}"
                )
                response.raise_for_status()
                stats = response.json()
                
                self._stats_cache[cache_key] = stats
                return stats
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch stats for {season} week {week}: {e}")
            return {}

    async def get_player_stats(
        self,
        player_name: str,
        season: int = 2025,
        week: Optional[int] = None
    ) -> Optional[dict]:
        """
        Get stats for a specific player.
        
        If week is specified, returns that week's stats.
        If week is None, returns season totals (sum of all weeks).
        
        Returns a dict with the player's metadata + their stats,
        or None if the player isn't found.
        """
        player = self.find_player(player_name)
        if not player:
            return None
        
        pid = player["player_id"]
        
        if week:
            # Single week stats
            weekly = await self.get_weekly_stats(season, week)
            stats = weekly.get(pid, {})
            return {**player, "stats": stats, "week": week, "season": season}
        else:
            # Season totals: aggregate across all available weeks
            season_stats: Dict[str, float] = {}
            games_played = 0
            
            for w in range(1, 19):
                weekly = await self.get_weekly_stats(season, w)
                if pid in weekly:
                    games_played += 1
                    for stat_key, value in weekly[pid].items():
                        if isinstance(value, (int, float)):
                            season_stats[stat_key] = season_stats.get(stat_key, 0) + value
            
            return {
                **player,
                "stats": season_stats,
                "games_played": games_played,
                "season": season,
                "week": "season_total"
            }

    async def get_projections(self, season: int, week: int) -> Dict[str, dict]:
        """Fetch weekly projections from Sleeper."""
        cache_key = (season, week, "proj")
        if cache_key in self._projections_cache:
            return self._projections_cache[cache_key]
        
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{SLEEPER_BASE_URL}/projections/nfl/regular/{season}/{week}"
                )
                response.raise_for_status()
                projections = response.json()
                
                self._projections_cache[cache_key] = projections
                return projections
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch projections: {e}")
            return {}

    # ─────────────────────────────────────────────────────────
    # Fantasy Point Calculations
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def calculate_fantasy_points(stats: dict, scoring: str = "ppr") -> float:
        """
        Calculate fantasy points from a stat line.
        
        This is a critical function — the scoring format determines
        how raw stats translate to fantasy value. The differences are:
        
        PPR:      +1.0 point per reception (favors pass-catchers)
        Half PPR: +0.5 point per reception (balanced)
        Standard: +0.0 point per reception (favors volume rushers)
        
        All other scoring is the same across formats.
        """
        points = 0.0
        
        # Passing
        points += stats.get("pass_yd", 0) * 0.04      # 1 point per 25 yards
        points += stats.get("pass_td", 0) * 4.0        # 4 points per passing TD
        points += stats.get("pass_int", 0) * -2.0      # -2 per interception
        points += stats.get("pass_2pt", 0) * 2.0       # 2 per passing 2pt conversion
        
        # Rushing
        points += stats.get("rush_yd", 0) * 0.1        # 1 point per 10 yards
        points += stats.get("rush_td", 0) * 6.0        # 6 points per rushing TD
        points += stats.get("rush_2pt", 0) * 2.0
        points += stats.get("fum_lost", 0) * -2.0      # -2 per fumble lost
        
        # Receiving
        points += stats.get("rec_yd", 0) * 0.1         # 1 point per 10 yards
        points += stats.get("rec_td", 0) * 6.0         # 6 points per receiving TD
        points += stats.get("rec_2pt", 0) * 2.0
        
        # The scoring format difference — receptions
        receptions = stats.get("rec", 0)
        if scoring == "ppr":
            points += receptions * 1.0
        elif scoring == "half_ppr":
            points += receptions * 0.5
        # standard: no points for receptions
        
        # Bonus for big games (common in many leagues)
        if stats.get("rush_yd", 0) >= 100:
            points += 3.0  # 100-yard rushing bonus
        if stats.get("rec_yd", 0) >= 100:
            points += 3.0  # 100-yard receiving bonus
        if stats.get("pass_yd", 0) >= 300:
            points += 3.0  # 300-yard passing bonus
        
        return round(points, 2)

    # ─────────────────────────────────────────────────────────
    # Formatting Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def format_player_summary(player: dict) -> str:
        """Format a player dict as a readable markdown summary."""
        name = player.get("full_name", "Unknown")
        pos = player.get("position", "?")
        team = player.get("team", "FA")
        age = player.get("age", "?")
        exp = player.get("years_exp", "?")
        injury = player.get("injury_status", None)
        
        summary = f"**{name}** ({pos} — {team})"
        summary += f"\nAge: {age} | Experience: {exp} years"
        
        if injury:
            body_part = player.get("injury_body_part", "undisclosed")
            summary += f"\n⚠️ Injury: {injury} ({body_part})"
        
        return summary

    @staticmethod
    def format_stats_line(stats: dict, scoring: str = "ppr") -> str:
        """Format a stat line as a readable string."""
        parts = []
        
        # Passing stats
        if stats.get("pass_yd", 0) > 0:
            parts.append(
                f"Passing: {stats.get('pass_yd', 0)} yds, "
                f"{stats.get('pass_td', 0)} TD, "
                f"{stats.get('pass_int', 0)} INT"
            )
        
        # Rushing stats
        if stats.get("rush_att", 0) > 0 or stats.get("rush_yd", 0) > 0:
            parts.append(
                f"Rushing: {stats.get('rush_att', 0)} att, "
                f"{stats.get('rush_yd', 0)} yds, "
                f"{stats.get('rush_td', 0)} TD"
            )
        
        # Receiving stats
        if stats.get("rec", 0) > 0 or stats.get("rec_yd", 0) > 0:
            parts.append(
                f"Receiving: {stats.get('rec', 0)} rec, "
                f"{stats.get('rec_yd', 0)} yds, "
                f"{stats.get('rec_td', 0)} TD"
            )
        
        if not parts:
            return "No stats recorded"
        
        return " | ".join(parts)

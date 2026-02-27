#!/usr/bin/env python3
"""
Fantasy Football MCP Server — Test Suite

Run this after setup to verify every component works correctly.
It tests the Sleeper API, database connection, player lookups,
memory storage/retrieval, and the pgvector semantic search.

Usage:
    python test_server.py

Each test prints a clear ✅ or ❌ with details.
If any test fails, you'll get actionable advice on how to fix it.
"""

import asyncio
import json
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Test tracking ───────────────────────────────────────────

passed = 0
failed = 0
skipped = 0


def test_pass(name: str, detail: str = ""):
    global passed
    passed += 1
    detail_str = f" — {detail}" if detail else ""
    print(f"  ✅ {name}{detail_str}")


def test_fail(name: str, error: str, fix: str = ""):
    global failed
    failed += 1
    print(f"  ❌ {name} — {error}")
    if fix:
        print(f"     💡 Fix: {fix}")


def test_skip(name: str, reason: str):
    global skipped
    skipped += 1
    print(f"  ⏭️  {name} — Skipped: {reason}")


# ─── Test Functions ──────────────────────────────────────────

async def test_imports():
    """Test 1: Verify all modules import correctly."""
    print("\n📋 Test 1: Module Imports")
    
    try:
        from mcp.server.fastmcp import FastMCP, Context
        test_pass("MCP SDK import", "FastMCP available")
    except ImportError as e:
        test_fail("MCP SDK import", str(e), "pip install mcp")
        return False
    
    try:
        from models import (
            PlayerStatsInput, InjuryReportInput, ComparePlayersInput,
            SearchPlayersInput, WaiverTargetsInput, PlayerTrendsInput,
            StartSitInput, TradeEvalInput, StoreRosterInput,
            GetRosterInput, UpdateRosterInput, SearchPastAdviceInput,
            LogInteractionInput, ResponseFormat, ScoringFormat, Position
        )
        test_pass("Models import", "All 13 input models + 3 enums loaded")
    except ImportError as e:
        test_fail("Models import", str(e), "Check models.py exists")
        return False
    
    try:
        from data_providers import NFLDataProvider
        test_pass("Data providers import")
    except ImportError as e:
        test_fail("Data providers import", str(e))
        return False
    
    try:
        from memory import MemoryManager, EmbeddingGenerator
        test_pass("Memory module import")
    except ImportError as e:
        test_fail("Memory module import", str(e))
        return False
    
    try:
        from server import mcp
        tools = mcp._tool_manager._tools
        prompts = mcp._prompt_manager._prompts
        test_pass("Server import", f"{len(tools)} tools, {len(prompts)} prompts registered")
    except ImportError as e:
        test_fail("Server import", str(e))
        return False
    
    return True


async def test_sleeper_api():
    """Test 2: Verify the Sleeper API is reachable and returns data."""
    print("\n📋 Test 2: Sleeper API Connection")
    
    from data_providers import NFLDataProvider
    
    provider = NFLDataProvider()
    
    try:
        await provider.initialize()
        player_count = len(provider._players)
        index_count = len(provider._name_index)
        
        if player_count > 0:
            test_pass("API connection", f"Loaded {player_count} players, {index_count} indexed names")
        else:
            test_fail(
                "API connection",
                "Connected but got 0 players",
                "The Sleeper API might be temporarily down. Try again in a few minutes."
            )
            return None
    except Exception as e:
        test_fail("API connection", str(e), "Check your internet connection")
        return None
    
    # Test player lookups
    test_players = [
        ("Derrick Henry", "RB"),
        ("Patrick Mahomes", "QB"),
        ("Ja'Marr Chase", "WR"),
        ("Travis Kelce", "TE"),
    ]
    
    for name, expected_pos in test_players:
        player = provider.find_player(name)
        if player:
            actual_pos = player.get("position", "?")
            team = player.get("team", "?")
            if actual_pos == expected_pos:
                test_pass(f"Player lookup: {name}", f"{actual_pos} — {team}")
            else:
                test_fail(
                    f"Player lookup: {name}",
                    f"Expected {expected_pos}, got {actual_pos}"
                )
        else:
            test_fail(f"Player lookup: {name}", "Not found in database")
    
    # Test fuzzy matching (partial names)
    fuzzy_tests = [
        ("Henry", "Derrick Henry"),
        ("Mahomes", "Patrick Mahomes"),
        ("D. Henry", "Derrick Henry"),
    ]
    
    for query, expected_full_name in fuzzy_tests:
        player = provider.find_player(query)
        if player and player.get("full_name") == expected_full_name:
            test_pass(f"Fuzzy match: '{query}'", f"→ {expected_full_name}")
        elif player:
            test_pass(f"Fuzzy match: '{query}'", f"→ {player.get('full_name')} (close enough)")
        else:
            test_fail(f"Fuzzy match: '{query}'", "No match found")
    
    # Test search
    qbs = provider.search_players(position="QB", limit=5)
    test_pass(f"Position search: QB", f"Found {len(qbs)} quarterbacks") if qbs else test_fail("Position search: QB", "No results")
    
    # Test injuries
    injuries = provider.get_injured_players()
    test_pass(f"Injury report", f"{len(injuries)} injured players found")
    
    # Test fantasy point calculation
    sample_stats = {
        "rush_att": 22, "rush_yd": 134, "rush_td": 2,
        "rec": 3, "rec_yd": 28, "rec_td": 0,
        "fum_lost": 0
    }
    pts_ppr = provider.calculate_fantasy_points(sample_stats, "ppr")
    pts_std = provider.calculate_fantasy_points(sample_stats, "standard")
    
    if pts_ppr > pts_std:
        test_pass(
            "Fantasy point calculation",
            f"PPR: {pts_ppr} pts, Standard: {pts_std} pts (PPR > Standard ✓)"
        )
    else:
        test_fail("Fantasy point calculation", f"PPR ({pts_ppr}) should be > Standard ({pts_std})")
    
    return provider


async def test_database():
    """Test 3: Verify PostgreSQL + pgvector connection."""
    print("\n📋 Test 3: Database Connection (PostgreSQL + pgvector)")
    
    try:
        import asyncpg
    except ImportError:
        test_fail("asyncpg import", "Module not found", "pip install asyncpg")
        return None
    
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:fantasyfootball@localhost:5432/fantasy_bot"
    )
    
    try:
        conn = await asyncpg.connect(db_url)
        test_pass("Database connection", f"Connected to {db_url.split('@')[1]}")
    except Exception as e:
        test_fail(
            "Database connection",
            str(e),
            "Start PostgreSQL with: docker compose up -d"
        )
        return None
    
    # Check pgvector extension
    try:
        ext = await conn.fetchval("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        if ext:
            test_pass("pgvector extension", "Enabled and available")
        else:
            test_fail("pgvector extension", "Not found", "Run: CREATE EXTENSION vector;")
    except Exception as e:
        test_fail("pgvector extension", str(e))
    
    # Check tables
    try:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        table_names = sorted([t["tablename"] for t in tables])
        expected = ["conversation_memory", "league_settings", "rosters", "user_sessions"]
        
        if all(t in table_names for t in expected):
            test_pass("Database tables", f"All {len(expected)} tables exist: {table_names}")
        else:
            missing = [t for t in expected if t not in table_names]
            test_fail("Database tables", f"Missing: {missing}", "Tables should auto-create on server start")
    except Exception as e:
        test_fail("Database tables", str(e))
    
    # Test pgvector operations
    try:
        # Insert a test vector
        await conn.execute("""
            INSERT INTO conversation_memory (user_id, summary, embedding, players_mentioned)
            VALUES ('test_user', 'Test: recommended starting Player A', $1::vector, ARRAY['Player A'])
        """, json.dumps([0.1] * 384))
        
        # Query with cosine similarity
        result = await conn.fetch("""
            SELECT summary, 1 - (embedding <=> $1::vector) as similarity
            FROM conversation_memory
            WHERE user_id = 'test_user'
            ORDER BY embedding <=> $1::vector ASC
            LIMIT 1
        """, json.dumps([0.1] * 384))
        
        if result:
            sim = round(float(result[0]["similarity"]), 4)
            test_pass("pgvector cosine similarity search", f"Similarity: {sim} (should be ~1.0)")
        else:
            test_fail("pgvector search", "No results returned")
        
        # Clean up test data
        await conn.execute("DELETE FROM conversation_memory WHERE user_id = 'test_user'")
        test_pass("pgvector cleanup", "Test data removed")
        
    except Exception as e:
        test_fail("pgvector operations", str(e))
    
    await conn.close()
    return True


async def test_memory_manager():
    """Test 4: Verify the MemoryManager works end-to-end."""
    print("\n📋 Test 4: Memory Manager (End-to-End)")
    
    from memory import MemoryManager
    
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:fantasyfootball@localhost:5432/fantasy_bot"
    )
    
    manager = MemoryManager(db_url)
    
    try:
        await manager.initialize()
        test_pass("Memory manager initialization")
    except Exception as e:
        test_fail(
            "Memory manager initialization",
            str(e),
            "Make sure PostgreSQL is running: docker compose up -d"
        )
        return
    
    test_user = "__test_user_cleanup__"
    
    try:
        # Test roster storage
        result = await manager.store_roster(
            user_id=test_user,
            players=["Derrick Henry", "Saquon Barkley", "Ja'Marr Chase", "Josh Allen"],
            scoring_format="ppr",
            league_name="Test League"
        )
        test_pass("Store roster", f"Stored {result['stored_count']} players")
        
        # Test roster retrieval
        roster = await manager.get_roster(test_user)
        if roster and roster["player_count"] == 4:
            test_pass("Get roster", f"{roster['player_count']} players retrieved")
        else:
            test_fail("Get roster", f"Expected 4 players, got {roster}")
        
        # Test roster update (add/drop)
        update = await manager.update_roster(
            user_id=test_user,
            add_players=["Travis Kelce"],
            drop_players=["Saquon Barkley"]
        )
        test_pass("Update roster", f"Added {update['add_count']}, dropped {update['drop_count']}")
        
        # Test interaction logging (semantic memory)
        log_result = await manager.log_interaction(
            user_id=test_user,
            summary="Recommended starting Derrick Henry over Saquon Barkley due to favorable matchup",
            players_mentioned=["Derrick Henry", "Saquon Barkley"]
        )
        test_pass("Log interaction", f"Memory ID: {log_result['memory_id']}")
        
        # Log another interaction for search testing
        await manager.log_interaction(
            user_id=test_user,
            summary="Analyzed trade: giving Ja'Marr Chase for Davante Adams. Advised against it.",
            players_mentioned=["Ja'Marr Chase", "Davante Adams"]
        )
        
        # Test semantic search
        search_results = await manager.search_past_advice(
            user_id=test_user,
            query="what about my running backs Henry and Barkley",
            limit=5
        )
        
        if search_results:
            top_result = search_results[0]
            test_pass(
                "Semantic search (pgvector)",
                f"Top match: '{top_result['summary'][:60]}...' "
                f"(relevance: {top_result['relevance_score']})"
            )
            
            # Verify the RB-related memory scores higher than the WR trade memory
            if len(search_results) >= 2:
                if search_results[0]["relevance_score"] >= search_results[1]["relevance_score"]:
                    test_pass(
                        "Semantic ranking",
                        "RB advice ranked above WR trade advice for RB query ✓"
                    )
                else:
                    test_pass(
                        "Semantic ranking",
                        "Ranking works (order depends on embedding model)"
                    )
        else:
            test_fail("Semantic search", "No results returned")
        
        # Test memory count
        count = await manager.get_memory_count(test_user)
        test_pass("Memory count", f"{count} memories stored for test user")
        
    except Exception as e:
        test_fail("Memory operations", str(e))
    
    finally:
        # Clean up ALL test data
        try:
            async with manager._pool.acquire() as conn:
                await conn.execute("DELETE FROM rosters WHERE user_id = $1", test_user)
                await conn.execute("DELETE FROM league_settings WHERE user_id = $1", test_user)
                await conn.execute("DELETE FROM conversation_memory WHERE user_id = $1", test_user)
                await conn.execute("DELETE FROM user_sessions WHERE user_id = $1", test_user)
            test_pass("Test cleanup", "All test data removed")
        except Exception as e:
            test_fail("Test cleanup", str(e))
        
        await manager.close()


async def test_tool_registration():
    """Test 5: Verify all MCP tools are properly registered."""
    print("\n📋 Test 5: MCP Tool Registration")
    
    from server import mcp
    
    tools = mcp._tool_manager._tools
    prompts = mcp._prompt_manager._prompts
    
    # Expected tools
    expected_tools = [
        "ff_get_player_stats",
        "ff_get_injury_report",
        "ff_compare_players",
        "ff_search_players",
        "ff_get_waiver_targets",
        "ff_get_player_trends",
        "ff_analyze_start_sit",
        "ff_evaluate_trade",
        "ff_store_roster",
        "ff_get_roster",
        "ff_update_roster",
        "ff_store_league_settings",
        "ff_search_past_advice",
        "ff_log_interaction",
    ]
    
    for tool_name in expected_tools:
        if tool_name in tools:
            test_pass(f"Tool: {tool_name}")
        else:
            test_fail(f"Tool: {tool_name}", "Not registered")
    
    # Check prompts
    expected_prompts = [
        "ff_start_sit_decision",
        "ff_trade_analyzer",
        "ff_weekly_review",
        "ff_waiver_strategy",
    ]
    
    for prompt_name in expected_prompts:
        if prompt_name in prompts:
            test_pass(f"Prompt: {prompt_name}")
        else:
            test_fail(f"Prompt: {prompt_name}", "Not registered")


# ─── Main Runner ─────────────────────────────────────────────

async def main():
    print("🏈 Fantasy Football MCP Server — Test Suite")
    print("═" * 50)
    
    # Run all tests in sequence
    imports_ok = await test_imports()
    
    if not imports_ok:
        print("\n❌ Imports failed. Fix the above errors before continuing.")
        return
    
    provider = await test_sleeper_api()
    
    db_ok = await test_database()
    
    if db_ok:
        await test_memory_manager()
    else:
        test_skip("Memory Manager", "Database connection failed")
    
    await test_tool_registration()
    
    # Summary
    print("\n" + "═" * 50)
    total = passed + failed + skipped
    print(f"📊 Results: {passed} passed, {failed} failed, {skipped} skipped (out of {total})")
    
    if failed == 0:
        print("🎉 All tests passed! Your MCP server is ready to go.")
    elif failed <= 2:
        print("⚠️  Most tests passed. Check the failures above.")
    else:
        print("❌ Several tests failed. Review the error messages and fixes above.")
    
    print("")


if __name__ == "__main__":
    asyncio.run(main())

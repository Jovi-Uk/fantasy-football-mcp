# 🏈 Fantasy Football MCP Server

**A Model Context Protocol server that gives AI agents real-time access to NFL player data, injury reports, matchup analysis, and persistent memory — powering the fantasy sports chatbot for our Gen AI class project.**

---

## What This Does

This MCP server is the data backbone of our fantasy football chatbot. When the Dify agent needs to answer a question like "Should I start Derrick Henry or Saquon Barkley this week?", it calls tools exposed by this server to get live data, compare players, and remember the user's roster across sessions.

The server exposes **17 tools**, **3 resources**, and **4 prompt templates** that any MCP-compatible client (like Dify) can use.

---

## Architecture

```
User asks a question
        │
        ▼
  Dify Agent (orchestrator)
        │
        ├──► RAG Knowledge Base (Hannah's docs: rankings, articles, CSVs)
        │
        └──► THIS MCP SERVER (http://your-ip:8000/mcp)
                │
                ├── Data Tools (Sleeper API → live NFL stats)
                │     ff_get_player_stats
                │     ff_get_injury_report  
                │     ff_compare_players
                │     ff_search_players
                │     ff_get_waiver_targets
                │     ff_get_player_trends
                │
                ├── Analysis Tools (reasoning + calculations)
                │     ff_analyze_start_sit
                │     ff_evaluate_trade
                │
                └── Memory Tools (PostgreSQL + pgvector)
                      ff_store_roster
                      ff_get_roster
                      ff_update_roster
                      ff_store_league_settings
                      ff_search_past_advice    ← pgvector semantic search
                      ff_log_interaction       ← stores embeddings
```

---

## Quick Start (My Machine — Server Host)

### Prerequisites

You need Docker (for PostgreSQL + pgvector) and Python 3.10+.

### Step 1: Start the database

```bash
cd fantasy-football-mcp
docker compose up -d
```

This starts PostgreSQL with pgvector on port 5432. The database schema is auto-created on first run.

### Step 2: Install Python dependencies

```bash
pip install -r requirements.txt

# OPTIONAL but recommended: better semantic memory
pip install sentence-transformers
```

### Step 3: Run the server

For local testing (STDIO transport):
```bash
python server.py
```

For team access (HTTP transport — use this so Dify can connect):
```bash
python server.py --transport http --port 8000
```

The server will log its startup progress and be ready when you see:
```
🏈 Server ready! Tools, resources, and prompts are available.
```

---

## For Teammates: Connecting Dify to This MCP Server

### Person 3 (Dify Workflow Builder) — How to Add the MCP Server

Once the server is running with HTTP transport, connect it in Dify:

1. Go to your Dify workspace → **Tools** (or **Tool Providers**)
2. Click **Add Tool Provider** → select **MCP**
3. Enter the server URL: `http://<server-host-ip>:8000/mcp`
   - If running on my laptop on the same network: use my local IP (e.g., `http://192.168.1.42:8000/mcp`)
   - If running on a cloud server: use the public IP or domain
4. Dify will automatically discover all 17 tools, 3 resources, and 4 prompts
5. Enable the tools you want the agent to use

### Testing the Connection

After connecting, you can test any tool in Dify's tool testing panel. Try:
- `ff_search_players` with `position: "QB"` → should return a list of NFL quarterbacks
- `ff_get_injury_report` with no parameters → should return current league-wide injuries

---

## Tool Reference

### Data Tools (Read-Only)

| Tool | What It Does | Example Use |
|------|-------------|-------------|
| `ff_get_player_stats` | Get rushing/passing/receiving stats | "How did Derrick Henry do last week?" |
| `ff_get_injury_report` | Current injury designations | "Who's injured on the Chiefs?" |
| `ff_compare_players` | Head-to-head stat comparison | "Compare Henry vs Barkley" |
| `ff_search_players` | Search/filter player database | "Show me all Bengals wide receivers" |
| `ff_get_waiver_targets` | Suggest available pickups | "Best RBs on waivers in a 12-team league" |
| `ff_get_player_trends` | Week-by-week performance trends | "Is Ja'Marr Chase trending up?" |

### Analysis Tools (Read-Only)

| Tool | What It Does | Example Use |
|------|-------------|-------------|
| `ff_analyze_start_sit` | Start/sit recommendation | "Start Henry or Barkley this week?" |
| `ff_evaluate_trade` | Trade value analysis | "Should I trade Chase for Adams?" |

### Memory Tools (Read/Write — Requires PostgreSQL)

| Tool | What It Does | Example Use |
|------|-------------|-------------|
| `ff_store_roster` | Save a user's roster | User shares their team |
| `ff_get_roster` | Retrieve stored roster | Before giving personalized advice |
| `ff_update_roster` | Add/drop players | "I just picked up Player X" |
| `ff_store_league_settings` | Save league config | "12-team PPR league" |
| `ff_search_past_advice` | Search past conversations (pgvector) | "What did you say about my RBs?" |
| `ff_log_interaction` | Store advice for future recall | After giving a recommendation |

---

## Data Sources

The primary data source is the **Sleeper API** (https://api.sleeper.app), which is free and requires no authentication. It provides all NFL player metadata, weekly stats, projections, and injury information.

The player database (~10,000+ players) is loaded into memory on server startup and cached locally for offline fallback.

---

## Project Structure

```
fantasy-football-mcp/
├── server.py              # Main MCP server — all tools, resources, prompts
├── models.py              # Pydantic input models for tool validation
├── data_providers.py      # Sleeper API integration + data caching
├── memory.py              # pgvector memory manager (roster + semantic search)
├── docker-compose.yml     # PostgreSQL + pgvector infrastructure
├── init_db.sql            # Database schema (auto-runs on first start)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── data/
    └── cache/             # Local cache for offline player data
```

---

## MCP Concepts Used (For the Presentation)

This project demonstrates several MCP capabilities worth highlighting:

**Tools**: 17 callable functions that the LLM agent can invoke to get data, perform analysis, and manage memory. Each tool has Pydantic-validated inputs and structured responses.

**Resources**: Browsable data endpoints (like REST APIs) that expose player profiles and team lists via URI templates.

**Prompts**: Pre-built prompt templates for common fantasy analysis patterns. The client fetches these and uses them as structured starting points.

**Lifespan Management**: Database connections and the player cache are initialized once at startup and shared efficiently across all requests.

**Tool Annotations**: Each tool declares metadata (read-only, destructive, idempotent) that helps the LLM make better decisions about when to use it.

**Streamable HTTP Transport**: The production transport that enables network access. Multiple clients can connect simultaneously.

---

## Troubleshooting

**"Memory system is not available"** — PostgreSQL isn't running. Start it with `docker compose up -d`.

**Player not found** — Try the full name (e.g., "Derrick Henry" not "Henry"). Use `ff_search_players` to find the exact name.

**Sleeper API errors** — The API may be rate-limited. The server will fall back to cached data automatically.

**Connection refused on port 8000** — Make sure the server is running with `--transport http`. Check your firewall allows port 8000.

---

## Team Workflow with YAML Exports

For sharing Dify configurations between team members, you can export your Dify app as a YAML file (Dify → Settings → Export). The MCP server connection URL will be in the YAML under the tool provider configuration. Make sure to update the IP address if importing on a different machine.

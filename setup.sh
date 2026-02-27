#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Fantasy Football MCP Server — macOS Setup Script
# ═══════════════════════════════════════════════════════════════
#
# This script sets up everything you need on your MacBook:
#   1. Verifies prerequisites (Python, Docker)
#   2. Creates a Python virtual environment
#   3. Installs all dependencies
#   4. Starts PostgreSQL + pgvector via Docker
#   5. Verifies the database connection
#   6. Runs a quick test of the MCP server
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After running this script:
#   - Activate the venv:  source .venv/bin/activate
#   - Run the server:     python server.py
#   - Or with HTTP:       python server.py --transport http --port 8000

set -e  # Exit on any error

echo "🏈 Fantasy Football MCP Server — Setup"
echo "═══════════════════════════════════════"
echo ""

# ─── Step 1: Check Prerequisites ────────────────────────────

echo "📋 Step 1: Checking prerequisites..."
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "   ✅ Python found: $PYTHON_VERSION"
else
    echo "   ❌ Python 3 not found. Install it from https://python.org"
    exit 1
fi

# Check Python version is 3.10+
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "   ❌ Python 3.10+ required. You have 3.$PYTHON_MINOR"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "   ✅ Docker found: $(docker --version)"
else
    echo "   ❌ Docker not found."
    echo "      Install Docker Desktop from: https://docker.com/products/docker-desktop/"
    echo "      After installing, run this script again."
    exit 1
fi

# Check Docker is running
if docker info &> /dev/null 2>&1; then
    echo "   ✅ Docker daemon is running"
else
    echo "   ⚠️  Docker is installed but not running."
    echo "      Please start Docker Desktop and run this script again."
    exit 1
fi

# Check if docker compose is available
if docker compose version &> /dev/null 2>&1; then
    echo "   ✅ Docker Compose found: $(docker compose version --short)"
    COMPOSE_CMD="docker compose"
elif docker-compose version &> /dev/null 2>&1; then
    echo "   ✅ Docker Compose found (legacy): $(docker-compose version --short)"
    COMPOSE_CMD="docker-compose"
else
    echo "   ❌ Docker Compose not found. It should come with Docker Desktop."
    exit 1
fi

echo ""

# ─── Step 2: Create Virtual Environment ─────────────────────

echo "📋 Step 2: Setting up Python virtual environment..."
echo ""

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "   ✅ Virtual environment created at .venv/"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate it
source .venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# ─── Step 3: Install Dependencies ───────────────────────────

echo "📋 Step 3: Installing Python dependencies..."
echo ""

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "   ✅ Core dependencies installed"

# Ask about optional sentence-transformers
echo ""
echo "   📦 Optional: sentence-transformers (~500MB download)"
echo "      This gives REAL semantic search in conversation memory."
echo "      Without it, a simpler hash-based fallback is used."
echo ""
read -p "   Install sentence-transformers? (y/n): " INSTALL_ST

if [ "$INSTALL_ST" = "y" ] || [ "$INSTALL_ST" = "Y" ]; then
    echo "   Installing sentence-transformers (this may take a few minutes)..."
    pip install sentence-transformers --quiet
    echo "   ✅ sentence-transformers installed"
else
    echo "   ⏭️  Skipping sentence-transformers (hash fallback will be used)"
fi

echo ""

# ─── Step 4: Start PostgreSQL + pgvector ─────────────────────

echo "📋 Step 4: Starting PostgreSQL + pgvector..."
echo ""

# Check if container is already running
if docker ps --filter "name=fantasy-pgvector" --format "{{.Names}}" | grep -q "fantasy-pgvector"; then
    echo "   ✅ PostgreSQL container already running"
else
    $COMPOSE_CMD up -d
    echo "   ⏳ Waiting for PostgreSQL to be ready..."
    
    # Wait for the database to be healthy
    RETRIES=0
    MAX_RETRIES=30
    until docker exec fantasy-pgvector pg_isready -U postgres &> /dev/null || [ $RETRIES -eq $MAX_RETRIES ]; do
        RETRIES=$((RETRIES + 1))
        sleep 1
    done
    
    if [ $RETRIES -eq $MAX_RETRIES ]; then
        echo "   ❌ PostgreSQL didn't start in time. Check: docker logs fantasy-pgvector"
        exit 1
    fi
    
    echo "   ✅ PostgreSQL + pgvector is running on port 5432"
fi

echo ""

# ─── Step 5: Verify Database Connection ─────────────────────

echo "📋 Step 5: Verifying database connection..."
echo ""

python3 -c "
import asyncio
import asyncpg

async def test_db():
    try:
        conn = await asyncpg.connect('postgresql://postgres:fantasyfootball@localhost:5432/fantasy_bot')
        
        # Check pgvector extension
        result = await conn.fetchval(\"SELECT extname FROM pg_extension WHERE extname = 'vector'\")
        if result:
            print('   ✅ pgvector extension is enabled')
        else:
            print('   ⚠️  pgvector extension not found, enabling...')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            print('   ✅ pgvector extension enabled')
        
        # Check tables exist
        tables = await conn.fetch(
            \"SELECT tablename FROM pg_tables WHERE schemaname = 'public'\"
        )
        table_names = [t['tablename'] for t in tables]
        print(f'   ✅ Database tables: {table_names}')
        
        await conn.close()
        return True
    except Exception as e:
        print(f'   ❌ Database connection failed: {e}')
        return False

result = asyncio.run(test_db())
if not result:
    exit(1)
"

echo ""

# ─── Step 6: Quick Server Test ──────────────────────────────

echo "📋 Step 6: Quick server verification..."
echo ""

python3 -c "
from server import mcp

tools = mcp._tool_manager._tools
prompts = mcp._prompt_manager._prompts

print(f'   ✅ MCP Server registered {len(tools)} tools:')
for name in sorted(tools.keys()):
    print(f'      • {name}')

print(f'   ✅ MCP Server registered {len(prompts)} prompt templates:')
for name in sorted(prompts.keys()):
    print(f'      • {name}')
"

echo ""

# ─── Step 7: Create .env file ───────────────────────────────

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✅ Created .env file from template"
else
    echo "   ✅ .env file already exists"
fi

echo ""

# ─── Done! ──────────────────────────────────────────────────

echo "═══════════════════════════════════════"
echo "🏈 Setup Complete!"
echo "═══════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment (if not already):"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run the server locally (STDIO — for Claude Desktop):"
echo "     python server.py"
echo ""
echo "  3. Or run with HTTP transport (for team/Dify access):"
echo "     python server.py --transport http --port 8000"
echo ""
echo "  4. To configure Claude Desktop, see: claude_desktop_config.json"
echo ""
echo "  5. To run the full test suite:"
echo "     python test_server.py"
echo ""

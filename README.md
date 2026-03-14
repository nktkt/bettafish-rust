# BettaFish Rust

A high-performance, multi-agent public opinion analysis system built in Rust.

BettaFish Rust is a modular deep-research engine that autonomously searches, analyzes, and synthesizes news from 30+ platforms into comprehensive professional reports. Users simply describe their research topic in natural language, and the system handles everything — from search query generation to iterative refinement to final report formatting.

## Architecture

The system is organized as a Cargo workspace with 10 specialized crates:

```
bettafish-rust/
├── crates/
│   ├── config/           # Configuration management (.env + env vars)
│   ├── common/           # Shared utilities (retry, text processing, forum reader)
│   ├── llm/              # OpenAI-compatible LLM client with streaming
│   ├── query-engine/     # Deep search agent (fully implemented)
│   ├── insight-engine/   # Local database sentiment analysis (interface)
│   ├── media-engine/     # Multimedia web search (interface)
│   ├── forum-engine/     # Multi-agent discussion orchestration (interface)
│   ├── report-engine/    # Template-based report generation (interface)
│   ├── mindspider/       # AI crawler cluster (interface)
│   └── app/              # Main application binary
```

## Key Features

- **AI-Driven Research Pipeline**: Automated report structure planning, iterative search-and-reflect loops, and professional formatting
- **6 Specialized Search Tools**: Basic search, deep analysis, 24h news, weekly news, image search, and date-range search via Tavily API
- **Reflection-Based Refinement**: Each section goes through multiple rounds of search and reflection to deepen analysis
- **Structured LLM Integration**: OpenAI-compatible API with streaming support, exponential backoff retry, and robust JSON parsing/repair
- **Forum-Aware Context**: Reads moderator summaries from multi-agent discussions to enrich analysis
- **Full State Persistence**: Save/load research state as JSON for resumable sessions
- **Async-First Design**: Built on Tokio for high-performance concurrent I/O

## QueryEngine — Full Implementation

The `query-engine` crate is a complete deep-search agent that orchestrates:

1. **Report Structure Generation** — LLM plans 3-5 sections for the research topic
2. **Initial Search & Summary** — For each section: select search tool, execute search, generate 800-1200 word summary
3. **Reflection Loop** — Identify gaps, search for missing information, merge findings (configurable iterations)
4. **Final Report Formatting** — Professional 10,000+ word Markdown report with tables, timelines, and source attribution

### Node Pipeline

| Node | Purpose |
|------|---------|
| `ReportStructureNode` | Plans report outline (3-5 sections) |
| `FirstSearchNode` | Selects search tool + generates query |
| `FirstSummaryNode` | Creates initial section summary |
| `ReflectionNode` | Identifies gaps and generates follow-up query |
| `ReflectionSummaryNode` | Merges new findings into existing section |
| `ReportFormattingNode` | Formats all sections into final Markdown report |

## Quick Start

### Prerequisites

- Rust 1.75+ (2021 edition)
- An OpenAI-compatible LLM API key (DeepSeek, OpenAI, etc.)
- A [Tavily API](https://www.tavily.com/) key for web search

### Configuration

Create a `.env` file in the project root:

```env
QUERY_ENGINE_API_KEY=your-llm-api-key
QUERY_ENGINE_BASE_URL=https://api.deepseek.com
QUERY_ENGINE_MODEL_NAME=deepseek-chat
TAVILY_API_KEY=your-tavily-api-key

# Optional tuning
MAX_REFLECTIONS=2
MAX_PARAGRAPHS=5
OUTPUT_DIR=reports
SAVE_INTERMEDIATE_STATES=true
```

### Build & Run

```bash
# Build
cargo build --release

# Run with a research query
./target/release/bettafish "artificial intelligence regulation trends 2025"

# Or run directly
cargo run --release -- "global chip competition analysis"
```

### Output

Reports are saved to the `reports/` directory as Markdown files:
- `deep_search_report_<topic>_<timestamp>.md` — The final report
- `state_<topic>_<timestamp>.json` — Full research state (if enabled)

## Crate Details

### `bettafish-config`
Loads configuration from `.env` files and environment variables. Supports all engine configurations (LLM keys, search APIs, database connections, tuning parameters).

### `bettafish-common`
Shared utilities:
- **Retry**: Exponential backoff with configurable profiles for LLM (6 retries, 60s initial), Search API (5 retries, 2s initial), and DB (5 retries, 1s initial)
- **Text Processing**: JSON tag cleaning, reasoning removal, incomplete JSON repair, content truncation
- **Forum Reader**: Reads agent discussion logs for context-aware analysis

### `bettafish-llm`
OpenAI-compatible LLM client with:
- Streaming and non-streaming modes
- UTF-8 safe stream concatenation
- Time context injection (current date/time in prompts)
- Configurable timeout (default: 1800s)

### Stub Engines (Interface Definitions)

These crates define trait interfaces for future implementation:

| Crate | Trait | Purpose |
|-------|-------|---------|
| `insight-engine` | `InsightEngine` | Local DB sentiment analysis with 22-language support |
| `media-engine` | `MediaEngine` | Multimodal web search (Bocha/Anspire APIs) |
| `forum-engine` | `ForumHost` | LLM-powered discussion moderator |
| `report-engine` | `ReportEngine` | HTML/PDF/Markdown rendering with IR schema |
| `mindspider` | `Spider` | Multi-platform crawler (7 social media platforms) |

## Testing

```bash
cargo test
```

## License

GPL-3.0

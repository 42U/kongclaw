# Zeraclaw Deployment Guide

How to package and distribute Zeraclaw so anyone can install and run it without pre-existing infrastructure.

---

## User-Facing Install Experience

```bash
# Install globally via npm
npm install -g kongclaw

# One-time setup — downloads dependencies, prompts for API key
kongclaw setup

# Start using it
kongclaw
```

That's it. No Docker, no manual model downloads, no config file editing.

---

## Infrastructure Dependencies

Zeraclaw requires three things users won't have out of the box:

| Dependency | What it is | Size | How we handle it |
|---|---|---|---|
| SurrealDB | Graph database for context storage | ~50MB binary | Auto-download platform binary |
| BGE-M3 GGUF | Local embedding model (1024-dim) | ~420MB | Auto-download from HuggingFace |
| bge-reranker-v2-m3 | Cross-encoder reranker (optional) | ~606MB | Manual download, auto-detected |
| Anthropic API key | LLM access | N/A | Interactive prompt during setup |

### SurrealDB — Standalone Binary Download

SurrealDB publishes standalone binaries for every platform. We download the correct one during setup rather than requiring Docker.

- Download URL pattern: `https://download.surrealdb.com/v{version}/surreal-v{version}.{platform}-{arch}.tgz`
- Platforms: `linux-amd64`, `linux-arm64`, `darwin-amd64`, `darwin-arm64`, `windows-amd64`
- Install location: `~/.kongclaw/bin/surreal`
- Zeraclaw manages the process lifecycle (start/stop/health check)

**Why not Docker?**
Docker is a heavy prerequisite that many users don't have installed. A single binary is simpler, faster to start, and has no dependency chain.

### BGE-M3 Embedding Model — Auto-Download

The GGUF quantized model is downloaded from HuggingFace on first setup.

- Source: `https://huggingface.co/compilade/bge-m3-GGUF/resolve/main/bge-m3-q4_k_m.gguf`
- Install location: `~/.kongclaw/models/bge-m3-q4_k_m.gguf`
- Show a progress bar during download (1.2GB takes a minute on decent internet)
- Skip if the file already exists

`node-llama-cpp` also provides `createModelDownloader()` which can be used as an alternative to a raw fetch.

### Anthropic API Key — Interactive Prompt

```
Enter your Anthropic API key: sk-ant-api03-...
Saved to ~/.kongclaw/config.json
```

Power users can skip the prompt by setting `ANTHROPIC_API_KEY` in their environment.

---

## Directory Structure

After setup, the user's machine has:

```
~/.kongclaw/
  bin/
    surreal              # SurrealDB binary
  models/
    bge-m3-q4_k_m.gguf   # Embedding model
  data/
    surreal/              # SurrealDB data files
  config.json             # API key, preferences
  surreal.pid             # PID file when SurrealDB is running
```

---

## CLI Commands

| Command | Description |
|---|---|
| `kongclaw` | Start the REPL (auto-starts SurrealDB if needed) |
| `kongclaw setup` | First-time setup: download deps, configure API key |
| `kongclaw start` | Start SurrealDB in the background |
| `kongclaw stop` | Stop the background SurrealDB process |
| `kongclaw status` | Show infrastructure health (SurrealDB, embeddings, API key) |
| `kongclaw reset` | Wipe graph data (keeps config and binaries) |

---

## `kongclaw setup` — Detailed Flow

```
kongclaw setup

[1/5] Detecting platform... linux-amd64
[2/5] Downloading SurrealDB v2.x.x... done (48MB)
      Saved to ~/.kongclaw/bin/surreal
[3/5] Downloading BGE-M3 embedding model...
      [========================================] 100% (1.2GB)
      Saved to ~/.kongclaw/models/bge-m3-q4_k_m.gguf
[4/5] Anthropic API key:
      Enter your key (sk-ant-...): ********
      Saved to ~/.kongclaw/config.json
[5/5] Verifying installation...
      SurrealDB:  started on ws://localhost:8042
      Schema:     applied (14 tables, 12 edges)
      Embeddings: loaded (1024 dimensions, 16ms/embed)
      API key:    valid

Ready! Run `kongclaw` to start.
```

---

## SurrealDB Process Management

Zeraclaw manages SurrealDB as a background process using a PID file.

### Start

```bash
~/.kongclaw/bin/surreal start \
  --bind 127.0.0.1:8042 \
  --user root --pass root \
  --log warn \
  file:~/.kongclaw/data/surreal
```

- Bind to localhost only (not exposed to network)
- Store the PID in `~/.kongclaw/surreal.pid`
- On `kongclaw` REPL start: check PID file, start if not running

### Stop

- Read PID from `~/.kongclaw/surreal.pid`
- Send SIGTERM, wait for clean shutdown
- Remove PID file

### Health Check

- Attempt WebSocket connection to `ws://localhost:8042/rpc`
- Timeout after 2 seconds
- Used by `kongclaw status` and auto-start logic

---

## Config File Format

`~/.kongclaw/config.json`:

```json
{
  "anthropicApiKey": "sk-ant-api03-...",
  "surreal": {
    "port": 8042,
    "user": "root",
    "pass": "root",
    "ns": "zera",
    "db": "memory"
  },
  "embedding": {
    "model": "bge-m3-q4_k_m",
    "dimensions": 1024
  }
}
```

All values have sensible defaults. The only required user input is the API key.

Environment variables override config file values:

| Env Var | Overrides |
|---|---|
| `ANTHROPIC_API_KEY` | `anthropicApiKey` |
| `SURREAL_URL` | Computed from `surreal.port` |
| `SURREAL_USER` / `SURREAL_PASS` | `surreal.user` / `surreal.pass` |
| `EMBED_MODEL_PATH` | Full path to a custom GGUF model |
| `ZERACLAW_MODEL` | Claude model (default: `claude-opus-4-6`) |

---

## Code Changes Required

### New Files

| File | Purpose |
|---|---|
| `src/setup.ts` | Setup command: platform detection, binary download, model download, API key prompt, verification |
| `src/lifecycle.ts` | SurrealDB process management: start, stop, health check, PID file |
| `src/download.ts` | Download utilities: fetch with progress bar, checksum verification, archive extraction |

### Modified Files

| File | Change |
|---|---|
| `src/config.ts` | Read from `~/.kongclaw/config.json` instead of `~/.surreal_env` and OpenClaw paths. Keep env var overrides. |
| `src/embeddings.ts` | Default model path to `~/.kongclaw/models/bge-m3-q4_k_m.gguf` |
| `src/index.ts` | Route subcommands (`setup`, `start`, `stop`, `status`, `reset`). Auto-start SurrealDB on REPL entry. |
| `package.json` | Add `postinstall` message pointing to `kongclaw setup`. Add subcommand routing. |

### package.json Updates

```json
{
  "bin": {
    "kongclaw": "./dist/index.js"
  },
  "scripts": {
    "postinstall": "echo '\\nRun `kongclaw setup` to complete installation.\\n'"
  }
}
```

---

## Distribution Channel

### Primary: npm

```bash
npm install -g kongclaw
```

**Why npm:**
- Target audience is developers (they have Node)
- `node-llama-cpp` compiles native bindings via npm postinstall — this just works
- Already have a `bin` field configured
- `npx kongclaw` works for try-before-install

### Future Channels

| Channel | Benefit | Complexity |
|---|---|---|
| **Homebrew tap** | `brew install kongclaw` — native feel on mac/linux | Medium — need a tap repo, formula that handles native deps |
| **Standalone binary** | No Node required, single download | Hard — `node-llama-cpp` native bindings make `pkg`/`bun compile` tricky |
| **Docker image** | Fully self-contained, everything bundled | Easy to build, but ironic given we're trying to avoid requiring Docker |

---

## Platform Support Matrix

| Platform | Architecture | SurrealDB | node-llama-cpp | Status |
|---|---|---|---|---|
| Linux | x86_64 | binary | compiles from source | Full support |
| Linux | ARM64 | binary | compiles from source | Full support |
| macOS | Apple Silicon | binary | compiles from source | Full support |
| macOS | Intel | binary | compiles from source | Full support |
| Windows | x86_64 | binary | compiles from source | Should work, needs testing |

### System Requirements

- **Node.js** 20+ (for `node-llama-cpp` compatibility)
- **RAM**: 2GB minimum (embedding model loads ~800MB)
- **Disk**: ~2GB (SurrealDB binary + embedding model + data)
- **C++ compiler**: Required for `node-llama-cpp` native build (cmake, gcc/clang)
  - Linux: `sudo apt install build-essential cmake`
  - macOS: Xcode Command Line Tools (usually already installed)
  - Windows: Visual Studio Build Tools

---

## Error Handling

The setup and runtime should handle these gracefully:

| Scenario | Behavior |
|---|---|
| No internet during setup | Clear error: "Download failed. Check your connection and retry `kongclaw setup`" |
| Disk full during model download | Detect, clean partial file, report needed space |
| Port 8042 already in use | Try next port (8043, 8044...), save to config |
| SurrealDB crashes mid-session | Graceful degradation — switch to recency-only context, warn user |
| Embedding model file corrupted | Detect on load failure, prompt to re-download with `kongclaw setup --redownload` |
| No C++ compiler for node-llama-cpp | npm install fails with clear message pointing to prerequisites |
| API key invalid | Detect on first API call, prompt to update with `kongclaw setup --key` |

---

## Security Considerations

- API key stored in `~/.kongclaw/config.json` with `0600` permissions (owner read/write only)
- SurrealDB binds to `127.0.0.1` only — not accessible from network
- No telemetry, no phoning home
- All embeddings computed locally — conversation content never leaves the machine (except to Anthropic for the LLM call)
- Add `~/.kongclaw/` to global gitignore recommendations in setup output

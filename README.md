# navidrome-mood-plugin

**Your music library, understood.** This plugin analyzes the actual audio in your Navidrome library using deep learning, then builds and maintains 13 mood playlists that evolve as your collection grows — automatically, on a schedule, with no manual tagging required.

Unlike metadata-based playlist tools, this one hears what you hear: a track that sounds aggressive gets scored aggressive, regardless of what genre tag it has. A quiet metal ballad in the wrong playlist? The three-layer scoring system catches it. A rare track that essentia misclassifies? Last.fm's crowd intelligence corrects it.

Works with any Subsonic-compatible client — Symfonium, Sublime Music, Ultrasonic, DSub, and others.

---

## What Makes This Different

Most music organization tools rely on metadata — genre tags you or someone else typed in. This plugin measures the music itself.

**Layer 1 — Deep audio analysis.** [Essentia-TensorFlow](https://essentia.upf.edu/) with Discogs-EffNet embeddings extracts mood, energy, danceability, and BPM from the raw audio waveform. Trained on 400,000+ tracks from the Discogs catalog.

**Layer 2 — Genre and BPM context correction.** Audio texture models have known blind spots. Drum & Bass scores near-zero on danceability because its 170 BPM patterns don't match training data. Metal ballads score high on relaxed because the audio texture is quiet. The genre boost layer adds 25+ keyword-matched corrections that restore what genre-blind audio analysis gets wrong.

**Layer 3 — Crowd-sourced cultural intelligence.** With a free Last.fm API key, each track's top crowd-sourced listener tags are fetched and used for a third adjustment pass. Nightwish crowd-tagged "symphonic metal" gets its relaxed score pushed down. A folk track tagged "chill" gets a relaxed boost. This layer adds cultural context that no audio model can infer from the waveform alone.

The result is 13 mood playlists that genuinely reflect how your music feels — not just what genre someone called it.

---

## What You Get

### 13 Mood Playlists, Auto-Built and Auto-Refreshed

**Simple moods** — a single score above a configurable threshold:

| Playlist | Signal | Default Threshold |
|----------|--------|-------------------|
| Happy Mix | mood_happy | 0.55 |
| Chill Mix | mood_relaxed | 0.40 |
| Energetic Mix | danceability | 0.60 |
| Melancholy Mix | mood_sad | 0.45 |
| Party Mix | mood_party | 0.55 |
| Aggressive Mix | mood_aggressive | 0.55 |

**Composite scenario playlists** — require a positive signal AND cap negative ones:

| Playlist | Requires | Caps | Sorted by |
|----------|----------|------|-----------|
| Study Mix | relaxed ≥ 0.40 | aggressive ≥ 0.45, party ≥ 0.50 | relaxed |
| Workout Mix | danceability ≥ 0.50 | relaxed ≥ 0.60, sad ≥ 0.50 | danceability |
| Sleep Mix | relaxed ≥ 0.30 | aggressive ≥ 0.30, party ≥ 0.35 | relaxed |
| Road Trip Mix | happy ≥ 0.35 | aggressive ≥ 0.40, sad ≥ 0.50 | happy |
| Cooking Mix | happy ≥ 0.35 | aggressive ≥ 0.45, sad ≥ 0.45 | happy |
| Dining Mix | relaxed ≥ 0.40 | aggressive ≥ 0.40 | relaxed |
| Background Mix | relaxed ≥ 0.35 | aggressive ≥ 0.50, party ≥ 0.55 | relaxed |

Genre exclusions are applied on top — hard blocklists ensure misclassified tracks never appear in calm mixes regardless of their audio scores. See [Genre Exclusions](#genre-exclusions).

### Mood-Aware Instant Mix

Replaces Navidrome's default Instant Mix with real mood-similarity matching. When you start an Instant Mix from any analyzed track, it calculates Euclidean distance across the full mood vector and returns the closest matches — not random tracks from the same genre, but tracks that actually feel similar.

### Fully Automated

- New tracks analyzed on a daily schedule (default: 2 AM)
- Playlists refreshed weekly (default: Sunday 3 AM) so they evolve as your library grows
- Uncertain tracks (low-confidence scores) automatically re-queued for re-analysis
- Optional random re-analysis percentage keeps scores fresh over time

---

## Quick Start

### 1. Start the Analyzer Service

The plugin needs an external service to run the audio analysis (essentia can't run inside WASM). A ready-to-use **multi-arch** Docker image (amd64 + arm64/Raspberry Pi) is provided.

Add it to your existing `docker-compose.yml`:

```yaml
services:
  navidrome:
    # ... your existing navidrome config ...
    networks:
      - mood

  mood-analyzer:
    image: ghcr.io/rflundgren/navidrome-mood-plugin:latest
    container_name: mood-analyzer
    restart: unless-stopped
    volumes:
      - /path/to/your/music:/music:ro
    # environment:                          # all optional — defaults shown
    #   MUSIC_DIR: /music                   # must match the container path above
    #   MAX_ANALYSIS_FILE_SIZE_MB: "200"
    #   FILE_ANALYSIS_TIMEOUT_SECONDS: "90"
    networks:
      - mood

networks:
  mood:
    driver: bridge
```

> **ARM64 users (Raspberry Pi):** The published image supports arm64. If you prefer to build locally, `cd analyzer-service && docker build --build-arg TARGETARCH=arm64 -t mood-analyzer .` will compile Essentia from source (~15–20 minutes on a Pi 5).

#### Analyzer Service Environment Variables

All optional — the container works out of the box with the defaults below. Only `MUSIC_DIR` needs attention, and only if you mount your library somewhere other than `/music` in the compose snippet above.

| Variable | Default | Description |
|----------|---------|--------------|
| `MUSIC_DIR` | `/music` | Container path the analyzer will read local files from for `/api/analysis/file`. Must match the right-hand side of the `volumes:` mount above — requests for anything outside this directory (including via symlinks) are rejected. |
| `MODELS_DIR` | `/app/models` | Where the Essentia/TensorFlow mood models live. The published image bakes them in at this path at build time; only relevant if you're customizing the Docker image. |
| `MAX_ANALYSIS_FILE_SIZE_MB` | `200` | Local files above this size are rejected before decoding, so an unexpectedly huge file can't balloon memory during analysis. |
| `FILE_ANALYSIS_TIMEOUT_SECONDS` | `90` | Wall-clock limit on a single `/api/analysis/file` request, mirroring the timeout `/api/analysis/url` already gets from its `ffmpeg` call. |

### 2. Install the Plugin

1. Download `mood-playlists.ndp` from [Releases](https://github.com/RFLundgren/navidrome-mood-plugin/releases)
2. Copy it to your Navidrome plugins directory: `<navidrome-data>/plugins/`
3. Restart Navidrome (or set `ND_PLUGINS_AUTORELOAD=true`)
4. Go to **Settings > Plugins > Mood Playlists** and approve permissions
5. Set **Analyzer Service URL** to `http://mood-analyzer:8000`
6. Set **Music Mount Path** to the path your music is mounted at in the analyzer container (e.g. `/music`)

### 3. Configure Agent Precedence (Instant Mix only)

If you use multiple Navidrome metadata agents, list `mood-playlists` first so Instant Mix uses this plugin's similarity matching:

```yaml
ND_AGENTS: mood-playlists,audiomuseai,lastfm,listenbrainz
```

### 4. Done

The plugin will start analyzing your library on its next scheduled run, or you can trigger an immediate analysis by temporarily setting the **Analysis Schedule** to fire in the next minute.

---

## Requirements

- **Navidrome** 0.62.0+
- **Docker** for the analyzer service
- Navidrome config:
  ```yaml
  ND_PLUGINS_ENABLED: "true"
  ND_PLUGINS_AUTORELOAD: "true"   # optional but recommended
  ```

---

## Last.fm Integration

Essentia scores audio texture. It cannot tell the difference between a quiet metal ballad and an ambient track — both have low dynamics and low brightness. Last.fm integration adds a third scoring layer using crowd-sourced listener tags, correcting for cultural context the audio models cannot infer.

### Setup

1. Get a free API key at [last.fm/api/account/create](https://www.last.fm/api/account/create)
2. In **Settings > Plugins > Mood Playlists**, paste your key into the **Last.fm API Key** field and save

New tracks analyzed after this will automatically include Last.fm lookups. Tracks already in your library need to be re-analyzed — see [Re-analyzing Your Library](#re-analyzing-your-library).

### What It Does

For each track, the analyzer fetches the top 10 listener tags from Last.fm and adjusts scores based on keyword matching:

| Tags | Effect |
|------|--------|
| metal, heavy, brutal, thrash | mood_aggressive +, mood_relaxed − |
| chill, relax, acoustic, ambient | mood_relaxed + |
| sad, melancholy, emotional | mood_sad + |
| dance, party, club, rave | danceability +, mood_party + |
| happy, uplifting, feel-good | mood_happy + |

Total Last.fm influence is capped at ±0.20 per score field so it blends with rather than overrides the essentia signal. If a track isn't found on Last.fm, analysis falls back to essentia + genre boosts — no scores are lost.

> **Note:** Last.fm only has data for widely scrobbled tracks. Rare releases, demos, and unreleased content may not be found and will be silently skipped.

### Tuning Boost Influence

Two multipliers let you control how much each correction layer contributes:

| Setting | Default | Effect |
|---------|---------|--------|
| Genre Boost Weight | 1.0 | 0.0 = genre boosts disabled, 2.0 = double influence |
| Last.fm Boost Weight | 1.0 | 0.0 = Last.fm disabled, 2.0 = double influence |

If your library is heavily genre-tagged and accurate, you may want to increase Genre Boost Weight. If Last.fm is returning poor tags for your catalog, reduce Last.fm Boost Weight or set it to 0. Both settings live in the Analyzer Service section of the plugin config.

---

## Genre Exclusions

Genre exclusions are hard blocklists applied during playlist generation. Tracks whose genre tag matches any keyword in a mix's exclusion list are ineligible for that mix, regardless of their mood scores.

This solves a fundamental limitation of audio texture models: a slow, quiet metal track can legitimately score high on `mood_relaxed` — the waveform really is quiet. Genre exclusions ensure it never appears in Chill or Sleep Mix.

**Default exclusions:**

| Mix | Excluded genres (substring match, case-insensitive) |
|-----|-----------------------------------------------------|
| Chill | metal, hard rock, punk, hardcore, industrial, grunge, thrash |
| Sleep | metal, hard rock, punk, hardcore, industrial, grunge, thrash, dance, techno, trance, house, edm, drum and bass |
| Study | metal, punk, hardcore, industrial |
| Dining | metal, hard rock, punk, hardcore, industrial |
| Background | metal, hard rock, punk, hardcore, industrial |
| Road Trip | metal, hardcore, industrial |

Each mix has its own config field (`chill_excluded_genres`, `sleep_excluded_genres`, etc.). Leave empty to use the defaults above. Enter a comma-separated list to override entirely.

**Genre migration:** If tracks were analyzed before genre exclusions were introduced, existing KV entries may lack genre data. Enable **Run Genre Migration** in the plugin settings to backfill genre from Navidrome into all existing entries — no re-analysis required. Disable it again after logs show `Genre migration complete`.

---

## Re-analyzing Your Library

The plugin analyzes each track once and caches the result. To re-analyze your entire library — for example after adding a Last.fm API key, or after updating the analyzer Docker image with improved models — use the **Force Re-analyze Entire Library** toggle:

1. In plugin settings, enable **Force Re-analyze Entire Library**
2. Set **Analysis Schedule** to fire in the next minute (e.g. `36 14 * * *` if it's currently 14:35 UTC)
3. Save — you'll see `Queued XXXX tracks` in the Navidrome logs shortly
4. **Important:** once queuing starts, restore **Analysis Schedule** to `0 2 * * *` and disable **Force Re-analyze Entire Library** — leaving it enabled will re-analyze everything on every subsequent run

Monitor progress:
```bash
docker logs navidrome -f | grep "mood-playlists"
```

Watch for `Reached end of library` to confirm completion. With a large library (10,000+ tracks) at 2 workers, expect several hours. After analysis completes, trigger a playlist refresh by temporarily setting **Playlist Refresh Schedule** to the next minute, then restore it.

---

## How It Works

```
┌─────────────────────────────────────┐     ┌──────────────────────────────────┐
│      Navidrome + mood-playlists      │     │       mood-analyzer service        │
│                                     │     │      (essentia-tensorflow)         │
│  Scheduler: daily scan              │────>│                                    │
│  → queue unanalyzed tracks          │ HTTP│  1. Extract audio features         │
│                                     │     │     BPM, energy, mood scores       │
│  Task executor: per track           │     │                                    │
│  → POST stream URL to analyzer      │     │  2. Genre + BPM context boosts     │
│  → store scores in KVStore          │     │     25+ keyword corrections        │
│                                     │     │                                    │
│  Scheduler: weekly refresh          │     │  3. Last.fm tag boosts (optional)  │
│  → query KVStore scores             │     │     crowd-sourced cultural context │
│  → build 13 mood playlists          │     │                                    │
│  → write via Subsonic API           │     │  → return final mood scores        │
│                                     │     └──────────────────────────────────┘
│  Instant Mix hook                   │
│  → Euclidean distance on mood       │
│    vectors → closest matches        │
└─────────────────────────────────────┘
```

The analyzer service receives a stream URL from the plugin, uses `ffmpeg` to download just the first 30 seconds of audio to a temp WAV, runs essentia-tensorflow inference, applies the two correction layers, and returns the final scores. The plugin stores them in its KVStore keyed by track ID and uses them for all subsequent playlist and Instant Mix operations.

---

## Configuration

All settings are in Navidrome → **Settings > Plugins > Mood Playlists**. Full reference:

| Setting | Default | Description |
|---------|---------|-------------|
| Navidrome URL | `http://navidrome:4533` | Internal URL of your Navidrome server |
| Analyzer Service URL | `http://mood-analyzer:8000` | URL of the mood analyzer service |
| Music Mount Path | `/music` | Path where music is mounted in the analyzer container |
| Auto-Analyze New Tracks | `true` | Scan for and analyze new tracks on schedule |
| Analysis Schedule | `0 2 * * *` | Cron expression (default: 2 AM daily) |
| Re-analyze Uncertain | `true` | Re-queue tracks with low-confidence scores |
| Re-analyze Percent | `0` | % of library to randomly re-analyze each cycle (0–20) |
| Re-analysis Schedule | `0 4 1 * *` | Cron for the dedicated re-analysis pass |
| Playlist Refresh Schedule | `0 3 * * 0` | Cron expression (default: Sunday 3 AM) |
| Tracks per Playlist | `30` | Number of tracks in each mood playlist |
| Similar Songs Count | `20` | Tracks returned for Instant Mix |
| Max Tracks per Artist | `3` | Per-artist cap per playlist (0 = no limit) |
| Max Analysis Workers | `2` | Concurrent analysis tasks (reduce to 1 on low-end hardware) |
| Playlist Variation Pool | `3` | Pool multiplier for weekly variety (1–10) |
| Happy Threshold | `0.55` | Minimum score (0–1) for happy classification |
| Chill Threshold | `0.40` | Minimum score for chill/relaxed |
| Energetic Threshold | `0.60` | Minimum score for energetic/danceable |
| Party Threshold | `0.55` | Minimum score for party |
| Melancholy Threshold | `0.45` | Minimum score for sad/melancholy |
| Aggressive Threshold | `0.55` | Minimum score for aggressive |
| Show Dates in Playlist Names | `true` | Append generation dates to playlist titles |
| Add Creation Dates to Playlists | `false` | Sync creation dates to all non-plugin playlists |
| Creation Date Sync Schedule | `0 5 * * *` | Cron for the creation date sync task |
| Run Genre Migration | `false` | One-time backfill of genre data into existing analyzed tracks |
| Genre Migration Schedule | `0 1 * * *` | Cron for the genre migration pass |
| Last.fm API Key | _(empty)_ | Free API key for crowd-sourced tag boosts |
| Genre Boost Weight | `1.0` | Multiplier for genre/BPM correction layer (0.0–2.0) |
| Last.fm Boost Weight | `1.0` | Multiplier for Last.fm tag correction layer (0.0–2.0) |
| Force Re-analyze Entire Library | `false` | Re-analyze every track on next run — disable after use |

---

## Building from Source

### Plugin (WASM)

```powershell
# Windows (PowerShell)
tinygo build -opt=2 -scheduler=none -no-debug -o plugin.wasm -target wasip1 -buildmode=c-shared .
Remove-Item mood-playlists.ndp -ErrorAction SilentlyContinue
Compress-Archive -Path plugin.wasm, manifest.json -DestinationPath mood-playlists.zip
Rename-Item mood-playlists.zip mood-playlists.ndp
```

```bash
# Linux / macOS
tinygo build -opt=2 -scheduler=none -no-debug -o plugin.wasm -target wasip1 -buildmode=c-shared .
zip mood-playlists.ndp plugin.wasm manifest.json
```

### Analyzer Service

```bash
cd analyzer-service
docker build -t mood-analyzer .
# ARM64: docker build --build-arg TARGETARCH=arm64 -t mood-analyzer .
```

---

## Documentation

For detailed troubleshooting, monitoring, per-field configuration reference, and the full Subsonic API integration guide, see **[HELP.md](HELP.md)**.

---

## Contributing

Contributions welcome. Ideas:

- [ ] Configurable thresholds for composite moods via the settings UI
- [ ] Per-user mood playlists
- [ ] "Mood of the day" rotating playlist

## License

GPL-3.0 — same as Navidrome.

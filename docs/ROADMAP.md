# Roadmap

Ideas for making submatch more useful to the community, roughly in order of impact.

---

## Community awareness

- **Demo GIF** — record a short terminal session showing submatch catching a real subtitle mismatch. Embed in the README.
- **Bazarr issue #1418** — comment on [morpheus65535/bazarr#1418](https://github.com/morpheus65535/bazarr/issues/1418); users there are hitting exactly the problem submatch solves.
- **r/selfhosted, r/DataHoarder, r/Plex** — post announcing submatch with the framing that no existing tool verifies subtitle content (ffsubsync/alass fix timing; Bazarr scores metadata; subgen regenerates). Optional follow-ups: r/jellyfin, r/kodi.

---

## Bazarr integration

[Bazarr](https://www.bazarr.media/) is the most widely used subtitle manager for Plex and Jellyfin. A post-download hook that calls submatch and rejects low-confidence subtitles would bring submatch into existing automated workflows without requiring users to change anything else. Bazarr already supports custom scripts triggered after each subtitle download.

## Watch mode

`submatch --watch /path/to/library` — monitor a directory tree for new video/subtitle pairs and automatically check them as they appear. Useful for fully automated setups where subtitles are downloaded in the background.

## Config file ✓

Implemented. `~/.config/submatch/config.toml` (or a local `./submatch.toml`) sets personal defaults for any flag. CLI flags always override config. See the README for the full list of configurable keys.

## Multiple audio track selection ✓

Implemented in v0.2.0. `--audio-track` accepts either an integer index (0-based) or a comma-separated language preference list (e.g. `--audio-track jp,en,pt`). If no track matches, falls back to track 0 with a warning.

## Embedded subtitle track matching ✓

Implemented. `submatch --embedded movie.mkv` extracts each internal subtitle stream via ffmpeg, scores it against the audio, and reports which tracks match. Works with directories too (`submatch --embedded /library/`). `--sub-lang` filters by track language tag. Exits 1 if any track fails.

## Library report

`submatch --report /path/to/library` — scan an entire media library and write an HTML or CSV summary of all checked pairs, their scores, and their states. Useful for auditing a large Plex or Jellyfin collection in one pass.

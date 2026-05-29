# Roadmap

Ideas for making submatch more useful to the community, roughly in order of impact.

---

## Bazarr integration

[Bazarr](https://www.bazarr.media/) is the most widely used subtitle manager for Plex and Jellyfin. A post-download hook that calls submatch and rejects low-confidence subtitles would bring submatch into existing automated workflows without requiring users to change anything else. Bazarr already supports custom scripts triggered after each subtitle download.

## Watch mode

`submatch --watch /path/to/library` — monitor a directory tree for new video/subtitle pairs and automatically check them as they appear. Useful for fully automated setups where subtitles are downloaded in the background.

## Config file

Support a `~/.config/submatch/config.toml` (or `~/.submatchrc`) for personal defaults: threshold, model, language, workers. Users running submatch regularly shouldn't need to repeat the same flags every time.

## Multiple audio track selection ✓

Implemented in v0.2.0. `--audio-track` accepts either an integer index (0-based) or a comma-separated language preference list (e.g. `--audio-track jp,en,pt`). If no track matches, falls back to track 0 with a warning.

## Embedded subtitle track matching ✓

Implemented. `submatch --embedded movie.mkv` extracts each internal subtitle stream via ffmpeg, scores it against the audio, and reports which tracks match. Works with directories too (`submatch --embedded /library/`). `--sub-lang` filters by track language tag. Exits 1 if any track fails.

## Library report

`submatch --report /path/to/library` — scan an entire media library and write an HTML or CSV summary of all checked pairs, their scores, and their states. Useful for auditing a large Plex or Jellyfin collection in one pass.

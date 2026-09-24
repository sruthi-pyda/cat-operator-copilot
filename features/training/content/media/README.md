# Lesson media

Drop lesson videos here and reference them from `lessons.yaml`:

```yaml
- lesson_id: L001
  video_url: "seatbelt.mp4"      # a file in this folder
  # or
  video_url: "https://..."        # a full URL
```

A filename is resolved against this folder. Anything starting with `http://` or
`https://` is passed straight to the player.

An empty `video_url` is fine — the Training Hub says "no video attached" rather
than rendering a broken player. A filename that does not exist here is reported
on screen instead of failing silently.

Nothing in this folder is committed except this README (see `.gitignore`), so
each machine keeps its own media.

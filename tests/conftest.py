import os
import shutil
import static_ffmpeg

# Prevent any test from accidentally sending events to the real Sentry project.
# Tests that call cli.main() directly (without patching telemetry.init) would
# otherwise initialize the SDK with the production DSN.
os.environ.setdefault("SUBMATCH_NO_TELEMETRY", "1")

if not shutil.which("ffmpeg"):
    static_ffmpeg.add_paths()

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,500
Hello, world.

2
00:00:05,000 --> 00:00:08,000
This is a test subtitle.
With two lines.

3
00:00:10,000 --> 00:00:12,000
Goodbye.
"""

SAMPLE_VTT = """\
WEBVTT

00:00:01.000 --> 00:00:03.500
Hello, world.

00:00:05.000 --> 00:00:08.000
This is a test subtitle.
With two lines.

00:00:10.000 --> 00:00:12.000
Goodbye.
"""

SAMPLE_ASS = """\
[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,Hello, world.
Comment: 0,0:00:00.00,0:00:00.00,Default,,0,0,0,,This line should be excluded.
Dialogue: 0,0:00:05.00,0:00:08.00,Default,,0,0,0,,This is a test subtitle.
Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,{\\i1}Goodbye.{\\i0}
"""

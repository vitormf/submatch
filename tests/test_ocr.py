from __future__ import annotations
from unittest.mock import patch, MagicMock


def _mock_proc(returncode=0):
    proc = MagicMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = returncode
    proc.pid = 12345
    return proc


def test_ocr_window_returns_text(tmp_path):
    """ocr_window extracts frames with ffmpeg and reads each with pytesseract."""
    frame = tmp_path / "frame0001.png"
    frame.touch()

    sub = tmp_path / "subtitle.sub"
    sub.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr.pytesseract.image_to_string", return_value="Hello world") as mock_tess:
        from submatch.ocr import ocr_window
        result = ocr_window(sub, start_ms=5000, duration_ms=30_000, lang="eng")

    assert result == "Hello world"
    mock_tess.assert_called_once()


def test_ocr_window_concatenates_multiple_frames(tmp_path):
    sub = tmp_path / "subtitle.sub"
    sub.touch()
    frame1 = tmp_path / "frame0001.png"
    frame2 = tmp_path / "frame0002.png"
    frame1.touch()
    frame2.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame1, frame2]), \
         patch("submatch.ocr.pytesseract.image_to_string", side_effect=["Hello", "world"]):
        from submatch.ocr import ocr_window
        result = ocr_window(tmp_path / "subtitle.sub", 0, 30_000, lang="eng")

    assert result == "Hello world"


def test_ocr_window_empty_when_no_frames(tmp_path):
    sub = tmp_path / "subtitle.sub"
    sub.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[]):
        from submatch.ocr import ocr_window
        result = ocr_window(sub, 0, 30_000, lang="eng")

    assert result == ""


def test_ocr_window_skips_empty_frame_text(tmp_path):
    sub = tmp_path / "subtitle.sub"
    sub.touch()
    frame = tmp_path / "frame0001.png"
    frame.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr.pytesseract.image_to_string", return_value="   "):
        from submatch.ocr import ocr_window
        result = ocr_window(sub, 0, 30_000, lang="eng")

    assert result == ""


def test_ocr_window_uses_osd_when_lang_none(tmp_path):
    """When lang=None, ocr_window calls _detect_lang_from_frame on the first frame."""
    sub = tmp_path / "subtitle.sup"
    sub.touch()
    frame = tmp_path / "frame0001.png"
    frame.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr._detect_lang_from_frame", return_value="jpn") as mock_osd, \
         patch("submatch.ocr.pytesseract.image_to_string", return_value="テスト") as mock_tess:
        from submatch.ocr import ocr_window
        ocr_window(sub, 0, 30_000, lang=None)

    mock_osd.assert_called_once_with(frame)
    mock_tess.assert_called_once_with(str(frame), lang="jpn")


def test_ocr_window_falls_back_to_eng_when_osd_returns_none(tmp_path):
    sub = tmp_path / "subtitle.sub"
    sub.touch()
    frame = tmp_path / "frame0001.png"
    frame.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr._detect_lang_from_frame", return_value=None), \
         patch("submatch.ocr.pytesseract.image_to_string", return_value="text") as mock_tess:
        from submatch.ocr import ocr_window
        ocr_window(sub, 0, 30_000, lang=None)

    mock_tess.assert_called_once_with(str(frame), lang="eng")


def test_ocr_window_ffmpeg_uses_correct_time_args(tmp_path):
    sub = tmp_path / "subtitle.sup"
    sub.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()) as mock_popen, \
         patch("submatch.ocr._frames_in_dir", return_value=[]):
        from submatch.ocr import ocr_window
        ocr_window(sub, start_ms=10_000, duration_ms=30_000, lang="eng")

    cmd = mock_popen.call_args[0][0]
    ss_idx = cmd.index("-ss")
    t_idx = cmd.index("-t")
    assert cmd[ss_idx + 1] == "10.0"
    assert cmd[t_idx + 1] == "30.0"


def test_is_tesseract_available_returns_true_when_installed():
    """is_tesseract_available returns True when tesseract binary is on PATH."""
    from submatch.ocr import is_tesseract_available
    with patch("submatch.ocr.pytesseract.get_tesseract_version", return_value="5.0.0"):
        assert is_tesseract_available() is True


def test_is_tesseract_available_returns_false_when_binary_missing():
    """is_tesseract_available returns False when the tesseract binary is not found."""
    import pytesseract
    from submatch.ocr import is_tesseract_available
    with patch("submatch.ocr.pytesseract.get_tesseract_version",
               side_effect=pytesseract.TesseractNotFoundError):
        assert is_tesseract_available() is False


def test_detect_lang_from_frame_latin_script(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    osd_output = "Page number: 0\nOrientation in degrees: 0\nScript: Latin\nScript confidence: 5.0\n"

    with patch("submatch.ocr.pytesseract.image_to_osd", return_value=osd_output):
        from submatch.ocr import _detect_lang_from_frame
        result = _detect_lang_from_frame(frame)

    assert result == "eng"


def test_detect_lang_from_frame_japanese_script(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()
    osd_output = "Script: Japanese\nScript confidence: 8.3\n"

    with patch("submatch.ocr.pytesseract.image_to_osd", return_value=osd_output):
        from submatch.ocr import _detect_lang_from_frame
        result = _detect_lang_from_frame(frame)

    assert result == "jpn"


def test_detect_lang_from_frame_returns_none_on_failure(tmp_path):
    frame = tmp_path / "frame.png"
    frame.touch()

    with patch("submatch.ocr.pytesseract.image_to_osd", side_effect=Exception("tesseract error")):
        from submatch.ocr import _detect_lang_from_frame
        result = _detect_lang_from_frame(frame)

    assert result is None


def test_ocr_window_pytesseract_runtime_error_returns_empty(tmp_path):
    """If pytesseract raises at runtime (e.g. tesseract binary missing), return ''."""
    sub = tmp_path / "subtitle.sub"
    sub.touch()
    frame = tmp_path / "frame0001.png"
    frame.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr.pytesseract.image_to_string", side_effect=Exception("TesseractNotFoundError")):
        from submatch.ocr import ocr_window
        result = ocr_window(sub, 0, 30_000, lang="eng")

    assert result == ""


# --- VOBSUB (.sub + .idx) path ---

def test_ocr_window_vobsub_uses_overlay_when_idx_present(tmp_path):
    """ocr_window dispatches to VOBSUB overlay path when .idx is present alongside .sub."""
    sub = tmp_path / "subtitle.sub"
    idx = tmp_path / "subtitle.idx"
    sub.touch()
    idx.touch()
    frame = tmp_path / "frame0001.png"
    frame.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()) as mock_popen, \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr._is_blank_frame", return_value=False), \
         patch("submatch.ocr.pytesseract.image_to_string", return_value="Hello"):
        from submatch.ocr import ocr_window
        result = ocr_window(sub, start_ms=5000, duration_ms=30_000, lang="eng")

    assert result == "Hello"
    cmd = mock_popen.call_args[0][0]
    assert "lavfi" in " ".join(cmd)
    assert "-filter_complex" in cmd


def test_ocr_window_vobsub_skips_blank_frames(tmp_path):
    """VOBSUB path filters blank (all-dark) frames before OCR."""
    sub = tmp_path / "subtitle.sub"
    idx = tmp_path / "subtitle.idx"
    sub.touch()
    idx.touch()
    frame1 = tmp_path / "frame0001.png"
    frame2 = tmp_path / "frame0002.png"
    frame1.touch()
    frame2.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame1, frame2]), \
         patch("submatch.ocr._is_blank_frame", side_effect=[True, False]), \
         patch("submatch.ocr.pytesseract.image_to_string", return_value="text") as mock_tess:
        from submatch.ocr import ocr_window
        ocr_window(sub, start_ms=0, duration_ms=30_000, lang="eng")

    assert mock_tess.call_count == 1  # only the non-blank frame


def test_ocr_window_vobsub_all_blank_returns_empty(tmp_path):
    """VOBSUB path returns '' when all frames are blank (no subtitle in time window)."""
    sub = tmp_path / "subtitle.sub"
    idx = tmp_path / "subtitle.idx"
    sub.touch()
    idx.touch()
    frame = tmp_path / "frame0001.png"
    frame.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()), \
         patch("submatch.ocr._frames_in_dir", return_value=[frame]), \
         patch("submatch.ocr._is_blank_frame", return_value=True):
        from submatch.ocr import ocr_window
        result = ocr_window(sub, start_ms=0, duration_ms=30_000, lang="eng")

    assert result == ""


def test_ocr_window_sub_without_idx_uses_direct_ffmpeg(tmp_path):
    """A .sub without a paired .idx skips the VOBSUB path and uses direct ffmpeg."""
    sub = tmp_path / "subtitle.sub"
    sub.touch()  # no .idx file

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()) as mock_popen, \
         patch("submatch.ocr._frames_in_dir", return_value=[]):
        from submatch.ocr import ocr_window
        ocr_window(sub, start_ms=0, duration_ms=30_000, lang="eng")

    cmd = mock_popen.call_args[0][0]
    assert "lavfi" not in " ".join(cmd)  # direct ffmpeg, not overlay


def test_ocr_window_vobsub_time_args(tmp_path):
    """VOBSUB path: -ss (input seek on subtitle) and -t (output duration limit) are correct."""
    sub = tmp_path / "subtitle.sub"
    idx = tmp_path / "subtitle.idx"
    sub.touch()
    idx.touch()

    with patch("submatch.ocr.subprocess.Popen", return_value=_mock_proc()) as mock_popen, \
         patch("submatch.ocr._frames_in_dir", return_value=[]):
        from submatch.ocr import ocr_window
        ocr_window(sub, start_ms=10_000, duration_ms=30_000, lang="eng")

    cmd = mock_popen.call_args[0][0]
    # -ss appears before the subtitle -i (input seek), -t appears after filter_complex (output limit)
    ss_idx = cmd.index("-ss")
    t_idx = cmd.index("-t")
    assert cmd[ss_idx + 1] == "10.0"
    assert cmd[t_idx + 1] == "30.0"
    # -ss must come before the subtitle input, -t after filter_complex
    sub_i_idx = cmd.index(str(sub))
    filter_idx = cmd.index("-filter_complex")
    assert ss_idx < sub_i_idx, "-ss must be input-level (before -i subtitle)"
    assert t_idx > filter_idx, "-t must be output-level (after -filter_complex)"


def test_is_blank_frame_dark_image(tmp_path):
    """_is_blank_frame returns True for an all-black image."""
    from PIL import Image
    from submatch.ocr import _is_blank_frame
    frame = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(frame)
    assert _is_blank_frame(frame) is True


def test_is_blank_frame_bright_pixel(tmp_path):
    """_is_blank_frame returns False for an image with a bright pixel (subtitle content)."""
    from PIL import Image
    from submatch.ocr import _is_blank_frame
    frame = tmp_path / "frame.png"
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    img.load()[50, 50] = (255, 255, 255)
    img.save(frame)
    assert _is_blank_frame(frame) is False


def test_is_blank_frame_returns_false_on_missing_file(tmp_path):
    """_is_blank_frame returns False (treat as non-blank) when PIL cannot open the file."""
    from submatch.ocr import _is_blank_frame
    assert _is_blank_frame(tmp_path / "nonexistent.png") is False

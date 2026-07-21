"""
OCR interface. process_image(file_path) -> raw_text

PaddleOCR 3.x rewrote its API (paddlex-based pipeline). show_log and some
old constructor args no longer exist, and .ocr() output format changed too.

On Windows, the new PIR executor + oneDNN combo has a known bug
(NotImplementedError: ConvertPirAttribute2RuntimeAttribute ... onednn).
Disabling mkldnn before the paddle backend initializes works around it.
"""
import os

# must be set before paddle/paddleocr is imported
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from paddleocr import PaddleOCR

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            # current (3.x) API — mkldnn disabled via env vars above to avoid
            # the Windows oneDNN/PIR NotImplementedError
            _ocr_engine = PaddleOCR(lang="en", enable_mkldnn=False)
        except TypeError:
            # enable_mkldnn not accepted at this constructor level on some versions
            _ocr_engine = PaddleOCR(lang="en")
        except Exception:
            # older (2.x) API fallback
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr_engine


def process_image(file_path: str) -> str:
    """
    Runs OCR on an image and returns raw extracted text, line by line.
    Handles blurry/low-quality images gracefully — returns whatever text
    was detected, even if partial, rather than crashing.
    """
    engine = get_ocr_engine()
    try:
        result = engine.predict(file_path)
    except AttributeError:
        # very old versions only have .ocr()
        result = engine.ocr(file_path, cls=True)

    if not result:
        return ""

    lines = []
    for page in result:
        # 3.x pipeline result: dict-like object with 'rec_texts'
        if hasattr(page, "get") and page.get("rec_texts"):
            lines.extend(page["rec_texts"])
        # 2.x list-of-lines format: [[box, (text, score)], ...]
        elif isinstance(page, list):
            for line in page:
                try:
                    lines.append(line[1][0])
                except (IndexError, TypeError):
                    continue

    return "\n".join(lines)
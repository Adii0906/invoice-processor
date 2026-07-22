"""
OCR interface. process_image(file_path) -> raw_text

Large phone-camera photos (often 3000-4000px on the long side) make
PaddleOCR much slower than necessary. Text detection accuracy doesn't
meaningfully improve past ~1600px on the long side for typical invoice
photos, so we downscale before running OCR. This is the single biggest
speed win available without changing the OCR engine itself.
"""
import os
from PIL import Image
import logging

# must be set before paddle/paddleocr is imported
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except Exception:
    PaddleOCR = None
    OCR_AVAILABLE = False
    logging.getLogger(__name__).warning("paddleocr or its dependencies are not available; OCR will be disabled")

_ocr_engine = None
MAX_DIMENSION = 1600


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        if not OCR_AVAILABLE:
            class _DummyOCR:
                def predict(self, *a, **k):
                    return []

                def ocr(self, *a, **k):
                    return []

            _ocr_engine = _DummyOCR()
            return _ocr_engine

        try:
            _ocr_engine = PaddleOCR(lang="en", enable_mkldnn=False)
        except TypeError:
            _ocr_engine = PaddleOCR(lang="en")
        except Exception:
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr_engine


def _downscale_if_needed(file_path: str) -> str:
    """Resizes the image in place (writes a temp copy) if it's larger than
    MAX_DIMENSION on the long side. Returns the path to use for OCR."""
    try:
        img = Image.open(file_path)
        w, h = img.size
        long_side = max(w, h)
        if long_side <= MAX_DIMENSION:
            return file_path

        scale = MAX_DIMENSION / long_side
        new_size = (int(w * scale), int(h * scale))
        img = img.convert("RGB").resize(new_size, Image.LANCZOS)

        resized_path = file_path + ".resized.jpg"
        img.save(resized_path, "JPEG", quality=90)
        return resized_path
    except Exception:
        # if resizing fails for any reason, fall back to the original file
        return file_path


def process_image(file_path: str) -> str:
    """
    Runs OCR on an image and returns raw extracted text, line by line.
    Handles blurry/low-quality images gracefully — returns whatever text
    was detected, even if partial, rather than crashing.
    """
    engine = get_ocr_engine()
    ocr_input_path = _downscale_if_needed(file_path)

    try:
        # Dummy engine returns empty list when paddleocr isn't installed
        result = engine.predict(ocr_input_path)
    except AttributeError:
        result = engine.ocr(ocr_input_path, cls=True)
    finally:
        if ocr_input_path != file_path and os.path.exists(ocr_input_path):
            os.remove(ocr_input_path)

    if not result:
        return ""

    lines = []
    for page in result:
        if hasattr(page, "get") and page.get("rec_texts"):
            lines.extend(page["rec_texts"])
        elif isinstance(page, list):
            for line in page:
                try:
                    lines.append(line[1][0])
                except (IndexError, TypeError):
                    continue

    return "\n".join(lines)
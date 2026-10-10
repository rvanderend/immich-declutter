"""Classification rules: is a photo/video a deletion candidate, and for which album?

Everything here is pure (no network) so it can be tested in isolation. The default
thresholds were calibrated on a real ~180k asset library (CLIP ViT-B-32__openai on
Immich thumbnails), measured as the share of hits in random samples:

- Memes: CLIP meme score >= 0.45 gave ~85% hits; a caption band at the top/bottom
  (OCR) combined with a CLIP caption score >= 0.5 ~40%; exactly 790 px (images saved
  from Facebook) ~90%; small (<= 640 px) images in a non-camera size ~85%.
- TikTok: OCR watermark (tiktok/reels/capcut/...) ~100%; 576x1024 is TikTok's
  download format.
- Screenshots: CLIP screenshot score >= 0.6.
- Receipts & barcodes: also applied to camera photos; 0.17% false positives on
  3000 random iPhone photos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MEMES = "memes"
TIKTOK = "tiktok"
SCREENSHOTS = "screenshots"
RECEIPTS = "receipts"
CATEGORIES = (MEMES, TIKTOK, SCREENSHOTS, RECEIPTS)

# --- CLIP prompts. Each group gets its own softmax over positives + negatives. ---

GENERAL_MEME = [
    "a meme", "an internet meme with text", "a funny image macro with caption text",
    "a cartoon", "a comic strip", "a quote with text on a plain background",
    "a motivational quote image", "a funny picture from the internet",
    "a screenshot of a social media post", "a screenshot of a tweet",
    "a clip art illustration", "a cartoon emoji", "a demotivational poster",
]
GENERAL_SCREEN = [
    "a screenshot of a phone", "a screenshot of a chat conversation", "a screenshot of a website",
    "a screenshot of an app", "a screenshot of a map", "a screenshot of a text message",
    "a screenshot of a game",
]
GENERAL_NEG = [
    "a photo of a person", "a selfie", "a family photo", "a photo of children", "a photo of a baby",
    "a group photo of friends", "a landscape photo", "a photo of a city street", "a photo of food",
    "a photo of a pet", "a photo of a room interior", "a photo of a garden", "a photo of a car",
    "a photo of a document", "a photo of a handwritten note", "a photo of a drawing made by a child",
    "a photo of a computer screen", "a photo of a party", "a holiday photo", "a photo at the beach",
    "a photo of a birthday cake", "a photo of flowers", "a blurry photo", "a dark photo",
    "a selfie with a funny face filter", "a photo of a building", "a photo of a sports event",
    "a photo of a wedding", "a photo of a birthday card on a table", "a home video of kids playing",
    "a vertical phone video", "a photo of a concert", "a photo of a poster", "a flyer",
    "an invitation card", "a photo of a product", "a photo of a ticket", "a photo of a tv screen",
]

CAPTION_POS = [
    "a meme with a caption above a photo", "a photo with bold black text at the top",
    "a funny meme picture with text",
]
CAPTION_NEG = [
    "a photo of a product", "a photo of an object", "a photo of a person", "a screenshot", "a map",
    "a flyer", "a photo of a document", "a photo of a receipt",
]

RECEIPT_RECEIPT = [
    "a photo of a receipt", "a cash register receipt", "a long paper store receipt",
    "a pin payment receipt",
]
RECEIPT_BARCODE = [
    "a barcode", "a QR code", "a shipping label with a barcode", "a parcel label",
    "a ticket with a barcode", "a loyalty card with a barcode", "a screenshot of a QR code",
    "a screenshot of a barcode", "a boarding pass",
]
RECEIPT_NEG = [
    "a photo of a document", "a letter", "a handwritten note", "a screenshot of a website",
    "a screenshot of a chat", "a photo of a person", "a family photo", "a photo of a child",
    "a photo of a baby", "a selfie", "a landscape", "a photo of food", "a photo of a product",
    "a photo of an object", "a photo of a tool", "a photo of electronics", "a photo of clothing",
    "a photo of a room", "a photo of furniture", "a blurry photo", "a dark photo",
    "a photo of a floor", "a photo of a wall", "a photo of a car", "a photo of a pet",
    "a price tag in a store", "a photo of a book page", "a menu", "a map", "a meme",
    "a photo of a computer screen", "a photo of a sign", "a photo of a package box", "a calendar",
    "a spreadsheet", "a poster", "a flyer", "a greeting card", "a drawing",
]

# name -> (positive groups, negatives). Every positive group yields its own score.
PROMPT_GROUPS = {
    "general": ([GENERAL_MEME, GENERAL_SCREEN], GENERAL_NEG),
    "caption": ([CAPTION_POS], CAPTION_NEG),
    "receipt": ([RECEIPT_RECEIPT, RECEIPT_BARCODE], RECEIPT_NEG),
}
SCORE_NAMES = {"general": ("meme", "screen"), "caption": ("caption",), "receipt": ("receipt", "barcode")}

# --- OCR keywords ---

TIKTOK_WORDS = ("tiktok", "reels", "capcut", "instagram", "snapchat")
MEME_SITE_WORDS = ("9gag", "imgflip", "dumpert", "geenstijl")
# English + Dutch till-receipt vocabulary (Immich's OCR reads both)
RECEIPT_WORDS = (
    "total", "subtotal", "receipt", "vat", "cash", "change", "terminal", "transaction",
    "maestro", "mastercard", "visa", "contactless",
    "totaal", "subtotaal", "btw", "kassabon", "bonnr", "pinnen", "betaald", "contant",
    "wisselgeld", "transactie", "kaartnr", "filiaal", "kassa",
)

# Common camera/webcam sizes; small images in any *other* size are almost always web images.
CAMERA_SMALL_SIZES = {
    (640, 480), (320, 240), (160, 120), (176, 144), (352, 288), (240, 240), (600, 600),
    (480, 360), (400, 300), (320, 320), (640, 360), (426, 320), (213, 160),
}
TIKTOK_SIZE = (1024, 576)


@dataclass(frozen=True)
class Thresholds:
    meme: float = 0.45
    meme_video: float = 0.6
    caption: float = 0.5
    screen: float = 0.6
    receipt_with_ocr: float = 0.3
    receipt: float = 0.85
    barcode: float = 0.75


@dataclass(frozen=True)
class Box:
    """OCR text box, coordinates normalised to 0..1."""

    top: float
    bottom: float
    text: str


def boxes_from_api(raw: list[dict]) -> list[Box]:
    return [
        Box(min(b["y1"], b["y2"]), max(b["y3"], b["y4"]), b.get("text") or "")
        for b in raw
        if b.get("isVisible", True)
    ]


def _tokens(boxes: list[Box]) -> list[str]:
    return re.findall(r"[a-z0-9]+", " ".join(b.text for b in boxes).lower())


def has_word(boxes: list[Box], words: tuple[str, ...]) -> str | None:
    """First keyword found. Short words (<5 chars) must match a whole token, longer ones may be part of one."""
    tokens = _tokens(boxes)
    joined = "".join(b.text for b in boxes).lower().replace(" ", "")
    for w in words:
        if len(w) < 5:
            if w in tokens:
                return w
        elif w in joined:
            return w
    return None


def caption_band(boxes: list[Box]) -> bool:
    """Large text only in a band at the top and/or bottom, hardly any other text (meme layout)."""
    big = [b for b in boxes if (b.bottom - b.top) >= 0.045 and len(b.text.strip()) >= 2]
    if not big:
        return False
    words = sum(len(b.text.split()) for b in big)
    top = [b for b in big if b.bottom <= 0.4]
    bottom = [b for b in big if b.top >= 0.6]
    other = len(boxes) - len(big)
    return words >= 4 and len(top) + len(bottom) == len(big) and other <= 2


def dims(asset: dict) -> tuple[int, int]:
    """(long side, short side)."""
    w, h = asset.get("width") or 0, asset.get("height") or 0
    return (max(w, h), min(w, h))


def is_web_size(asset: dict) -> str | None:
    long_, short = dims(asset)
    if not long_:
        return None
    if 790 in (long_, short):
        return "790px"
    if long_ <= 640 and (long_, short) not in CAMERA_SMALL_SIZES:
        return "small-web-size"
    return None


def has_camera(asset: dict) -> bool:
    return bool((asset.get("exifInfo") or {}).get("make"))


def needs_analysis(asset: dict, enabled: set[str]) -> bool:
    """Whether it is worth fetching thumbnail/CLIP/OCR at all."""
    if asset.get("type") == "VIDEO":
        return TIKTOK in enabled and not has_camera(asset)
    if has_camera(asset):
        return RECEIPTS in enabled
    return True


def classify(
    asset: dict,
    scores: dict[str, float],
    boxes: list[Box],
    enabled: set[str],
    t: Thresholds = Thresholds(),
) -> tuple[str | None, str]:
    """Returns (category or None, reason)."""
    camera = has_camera(asset)

    if asset.get("type") == "VIDEO":
        if camera or TIKTOK not in enabled:
            return None, ""
        if w := has_word(boxes, TIKTOK_WORDS):
            return TIKTOK, f"ocr:{w}"
        if dims(asset) == TIKTOK_SIZE:
            return TIKTOK, "576x1024"
        if scores.get("meme", 0) >= t.meme_video:
            return TIKTOK, f"meme={scores['meme']:.2f}"
        return None, ""

    if not camera and TIKTOK in enabled and has_word(boxes, ("tiktok",)):
        return TIKTOK, "ocr:tiktok"

    if RECEIPTS in enabled:
        r, b = scores.get("receipt", 0), scores.get("barcode", 0)
        if r >= t.receipt_with_ocr and (w := has_word(boxes, RECEIPT_WORDS)):
            return RECEIPTS, f"receipt={r:.2f}+ocr:{w}"
        if r >= t.receipt:
            return RECEIPTS, f"receipt={r:.2f}"
        if b >= t.barcode:
            return RECEIPTS, f"barcode={b:.2f}"

    if camera:
        return None, ""

    if MEMES in enabled:
        if scores.get("meme", 0) >= t.meme:
            return MEMES, f"meme={scores['meme']:.2f}"
        if w := has_word(boxes, MEME_SITE_WORDS):
            return MEMES, f"ocr:{w}"
        if caption_band(boxes) and scores.get("caption", 0) >= t.caption:
            return MEMES, f"caption-band+caption={scores['caption']:.2f}"
        if size := is_web_size(asset):
            return MEMES, size

    if SCREENSHOTS in enabled and scores.get("screen", 0) >= t.screen:
        return SCREENSHOTS, f"screen={scores['screen']:.2f}"

    return None, ""

import rules
from rules import Box, classify

ALL = set(rules.CATEGORIES)


def img(w=1200, h=1500, make=None, type_="IMAGE"):
    return {"type": type_, "width": w, "height": h, "exifInfo": {"make": make}}


# OCR of a real meme CLIP missed (meme=0.02): caption band at the top
SPEAKERS = [
    Box(0.017, 0.089, "knew"),
    Box(0.099, 0.181, "had a call.."),
    Box(0.101, 0.189, "you H"),
    Box(0.183, 0.279, "before your phone did"),
    Box(0.02, 0.09, "These speakers"),
]


def test_caption_band_meme():
    assert rules.caption_band(SPEAKERS)
    cat, reason = classify(img(), {"meme": 0.02, "caption": 0.56}, SPEAKERS, ALL)
    assert cat == rules.MEMES and reason.startswith("caption-band")


def test_caption_band_needs_clip():
    assert classify(img(), {"meme": 0.02, "caption": 0.2}, SPEAKERS, ALL)[0] is None


def test_text_all_over_is_no_band():
    boxes = [Box(y, y + 0.05, "lorem ipsum dolor") for y in (0.1, 0.45, 0.8)]
    assert not rules.caption_band(boxes)


def test_web_sizes():
    assert classify(img(790, 1073), {}, [], ALL) == (rules.MEMES, "790px")
    assert classify(img(400, 347), {}, [], ALL) == (rules.MEMES, "small-web-size")
    # old webcam/phone: standard size, not a candidate
    assert classify(img(640, 480), {}, [], ALL)[0] is None
    assert classify(img(1200, 1600), {}, [], ALL)[0] is None


def test_camera_photo_only_receipts():
    cam = img(make="Apple")
    assert classify(cam, {"meme": 0.99, "screen": 0.99}, [], ALL)[0] is None
    label = [Box(0.5, 0.52, "PostNL Pakket 3SABC123 Amsterdam 1012 AB retour afzender track trace")]
    assert classify(cam, {"barcode": 0.8}, label, ALL)[0] == rules.RECEIPTS
    # an airplane photo with a livery as its only text is not a barcode
    assert classify(cam, {"barcode": 0.88}, [Box(0.3, 0.4, "transavia")], ALL)[0] is None
    bon = [Box(0.5, 0.52, "AH Zwijndrecht melk 1,09 brood 2,49 kaas 5,99 TOTAAL 12,95 PIN")]
    assert classify(cam, {"receipt": 0.4}, bon, ALL)[0] == rules.RECEIPTS
    assert classify(cam, {"receipt": 0.4}, [Box(0.5, 0.52, "TOTAAL 12,95")], ALL)[0] is None
    assert classify(cam, {"receipt": 0.4}, [Box(0.5, 0.52, "hello")], ALL)[0] is None
    assert not rules.needs_analysis(cam, ALL - {rules.RECEIPTS})


def test_short_ocr_words_exact():
    assert rules.has_word([Box(0, 0.1, "BTW 21%")], rules.RECEIPT_WORDS) == "btw"
    assert rules.has_word([Box(0, 0.1, "abtwx")], ("btw",)) is None
    assert rules.has_word([Box(0, 0.1, "Sub totaal")], ("subtotaal",)) == "subtotaal"


def test_tiktok_video():
    # TikTok's download size alone is not enough: phones record personal videos in it too
    assert classify(img(576, 1024, type_="VIDEO"), {}, [], ALL)[0] is None
    vid2 = img(720, 1280, type_="VIDEO")
    assert classify(vid2, {}, [Box(0.9, 0.95, "TikTok @iemand")], ALL)[0] == rules.TIKTOK
    assert classify(vid2, {"meme": 0.3}, [], ALL)[0] is None
    assert classify(img(576, 1024, make="Apple", type_="VIDEO"), {}, [], ALL)[0] is None


def test_tiktok_watermark_on_image_is_meme():
    # The TikTok album only holds videos
    assert classify(img(), {}, [Box(0.9, 0.95, "TikTok")], ALL)[0] == rules.MEMES


def test_screenshot_and_disabled_categories():
    assert classify(img(), {"screen": 0.7}, [], ALL)[0] == rules.SCREENSHOTS
    assert classify(img(), {"screen": 0.7}, [], {rules.MEMES})[0] is None
    assert classify(img(790, 1073), {}, [], {rules.SCREENSHOTS})[0] is None


def test_priority_meme_before_screenshot():
    assert classify(img(), {"meme": 0.5, "screen": 0.9}, [], ALL)[0] == rules.MEMES


def test_boxes_from_api():
    raw = [{"x1": 0, "y1": 0.2, "x2": 1, "y2": 0.18, "x3": 1, "y3": 0.28, "x4": 0, "y4": 0.3,
            "text": "hoi", "isVisible": True}]
    assert rules.boxes_from_api(raw) == [Box(0.18, 0.3, "hoi")]


def test_english_receipt_words():
    assert rules.has_word([Box(0, 0.1, "SUBTOTAL 12.95")], rules.RECEIPT_WORDS) == "total"
    assert rules.has_word([Box(0, 0.1, "VAT 20%")], rules.RECEIPT_WORDS) == "vat"


def test_needs_ocr():
    t = rules.Thresholds()
    assert rules.needs_ocr(img(), {}, t)
    assert not rules.needs_ocr(img(make="Apple"), {"receipt": 0.1, "barcode": 0.5}, t)
    assert rules.needs_ocr(img(make="Apple"), {"barcode": 0.8}, t)


def test_generic_words_dropped():
    # 'cash'/'change' matched a toy cash register and a petrol pump
    assert rules.has_word([Box(0, 0.1, "CASH REGISTER change")], rules.RECEIPT_WORDS) is None

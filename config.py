"""Fixed research protocol; changing these values invalidates downstream signatures."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
VIEWPORT = {"width": 1280, "height": 800}
MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
BATCH_SIZE = 8
THREADS = 4
CATEGORIES = {
    "content": [
        "a screenshot of a normal website homepage with navigation and content",
        "a screenshot of a news, shopping or company website",
    ],
    "login": [
        "a screenshot of a login page asking for username and password",
        "a screenshot of a sign in or account registration form",
    ],
    "captcha": [
        "a screenshot of a CAPTCHA or robot verification challenge",
        "a screenshot saying verify you are human or checking your browser",
    ],
    "cookie": [
        "a screenshot of a website obscured by a large cookie consent dialog",
        "a screenshot of a privacy consent popup with accept and reject buttons",
    ],
    "for_sale": [
        "a screenshot of a parked domain for sale",
        "a screenshot advertising buy this domain name",
    ],
    "error": [
        "a screenshot of an access denied or forbidden error page",
        "a screenshot of a server error or website unavailable page",
    ],
}

HF_REPOSITORY = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
HF_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
WEIGHT_FILENAME = "open_clip_model.safetensors"
WEIGHT_SHA256 = "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"

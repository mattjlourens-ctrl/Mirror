"""Finding product photos on a retailer's page with standard HTML metadata.

Sources, most trusted first: schema.org Product data (JSON-LD), Open Graph and
Twitter card images, <link rel="image_src">, <img> elements, and finally image
URLs embedded in the page's scripts, which is where pages rendered with
JavaScript keep them. Obvious non-product images (logos, icons, payment
badges...) are skipped by name.
"""
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from backend.errors import AccessRestricted, FetchFailed

PRIORITY = {"json-ld": 0, "open-graph": 1, "twitter": 2, "link": 2, "img": 3, "embedded": 4}
SKIP_WORDS = re.compile(
    r"logo|icon|sprite|favicon|badge|payment|paypal|klarna|visa|mastercard|flag|avatar|"
    r"rating|stars?\b|banner|placeholder|spinner|loading|pixel|tracking|social|facebook|"
    r"instagram|pinterest|twitter|youtube|tiktok|qr[-_]?code|blank|spacer|swatch",
    re.IGNORECASE)
FRONT_WORDS = re.compile(r"front|frontal|\bfr\b|_f\.|-f\.|main|primary|hero", re.IGNORECASE)
IMAGE_EXTENSIONS = re.compile(r"\.(svg|gif|ico)(\?|$)", re.IGNORECASE)
MIN_DECLARED_SIDE = 150
# Absolute image URLs inside inline scripts or JSON, possibly with JSON escapes.
EMBEDDED_IMAGE_URL = re.compile(r'''https?:(?:\\?/){2}[^\s"'<>()]+?\.(?:jpe?g|png|webp)(?:\?[^\s"'<>()]*)?''',
                                re.IGNORECASE)



@dataclass(frozen=True)
class ImageReference:
    url: str
    source: str   # json-ld, open-graph, twitter, link or img
    order: int    # position on the page
    front_hint: bool


def _largest_from_srcset(srcset):
    best, best_width = None, -1
    for entry in srcset.split(","):
        pieces = entry.strip().split()
        if not pieces:
            continue
        width = 0
        if len(pieces) > 1 and pieces[1].endswith(("w", "x")):
            try:
                width = float(pieces[1][:-1]) * (1 if pieces[1].endswith("w") else 1000)
            except ValueError:
                width = 0
        if width > best_width:
            best, best_width = pieces[0], width
    return best


def _json_ld_images(node, found):
    """Collect image URLs of schema.org Product (or ProductGroup) objects."""
    if isinstance(node, list):
        for item in node:
            _json_ld_images(item, found)
        return
    if not isinstance(node, dict):
        return
    types = node.get("@type", [])
    types = types if isinstance(types, list) else [types]
    if any(str(kind) in ("Product", "ProductGroup", "IndividualProduct") for kind in types):
        images = node.get("image", [])
        for image in images if isinstance(images, list) else [images]:
            if isinstance(image, str):
                found.append(image)
            elif isinstance(image, dict):
                url = image.get("contentUrl") or image.get("url")
                if isinstance(url, str):
                    found.append(url)
    for key in ("@graph", "hasVariant", "mainEntity", "itemListElement"):
        if key in node:
            _json_ld_images(node[key], found)


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.base = None
        self.found = []          # (source, url, hint text)
        self._json_ld = None     # text of the <script type="application/ld+json"> being read

    def handle_starttag(self, tag, attrs):
        attrs = {name.lower(): (value or "") for name, value in attrs}
        if tag == "base" and attrs.get("href") and self.base is None:
            self.base = attrs["href"]
        elif tag == "meta":
            key = (attrs.get("property") or attrs.get("name") or "").lower()
            content = attrs.get("content", "")
            if key in ("og:image", "og:image:url", "og:image:secure_url"):
                self.found.append(("open-graph", content, ""))
            elif key in ("twitter:image", "twitter:image:src"):
                self.found.append(("twitter", content, ""))
        elif tag == "link" and "image_src" in attrs.get("rel", "").lower().split():
            self.found.append(("link", attrs.get("href", ""), ""))
        elif tag == "script" and attrs.get("type", "").lower() == "application/ld+json":
            self._json_ld = []
        elif tag == "img":
            hint = " ".join(attrs.get(name, "") for name in ("alt", "class", "id", "title"))
            for size in ("width", "height"):
                if attrs.get(size, "").isdigit() and int(attrs[size]) < MIN_DECLARED_SIDE:
                    return
            url = (attrs.get("data-zoom-image") or attrs.get("data-large_image")
                   or (_largest_from_srcset(attrs["srcset"]) if attrs.get("srcset") else None)
                   or (_largest_from_srcset(attrs["data-srcset"]) if attrs.get("data-srcset") else None)
                   or attrs.get("data-src") or attrs.get("src", ""))
            self.found.append(("img", url, hint))

    def handle_data(self, data):
        if self._json_ld is not None:
            self._json_ld.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._json_ld is not None:
            text, self._json_ld = "".join(self._json_ld), None
            try:
                images = []
                _json_ld_images(json.loads(text), images)
            except (ValueError, RecursionError):
                return
            self.found.extend(("json-ld", url, "") for url in images)


def find_product_images(html, page_url, limit):
    """Up to `limit` likely product photos, best first."""
    collector = _Collector()
    try:
        collector.feed(html)
        collector.close()
    except Exception:  # html.parser is lenient, but never let a bad page crash a request
        pass
    base = urljoin(page_url, collector.base) if collector.base else page_url
    found = collector.found + [("embedded", _unescape(match), "") for match in EMBEDDED_IMAGE_URL.findall(html)]

    references, seen = [], set()
    for order, (source, raw, hint) in enumerate(found):
        raw = raw.strip()
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(base, raw)
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            continue
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
        if url in seen or IMAGE_EXTENSIONS.search(parts.path) or SKIP_WORDS.search(parts.path + " " + hint):
            continue
        seen.add(url)
        references.append(ImageReference(url, source, order, bool(FRONT_WORDS.search(parts.path + " " + hint))))

    references.sort(key=lambda ref: (PRIORITY[ref.source], not ref.front_hint, ref.order))
    return references[:limit]


def _unescape(url):
    """Undo JSON string escaping in a URL found in script text."""
    url = url.replace("\\/", "/")
    url = re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), url)
    return url.rstrip("\\")


def check_robots(fetcher, page_url, user_agent):
    """Raise AccessRestricted if the site's robots.txt disallows fetching the page."""
    parts = urlsplit(page_url)
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        result = fetcher.get(robots_url, max_bytes=512 * 1024, accept="text/plain")
    except FetchFailed as error:
        if error.code == "too_large":
            return  # an unreadable robots.txt is treated like a missing one
        raise
    restricted = AccessRestricted(
        "This store doesn't allow automatic access to its pages. Save the product photo and upload it instead.")
    if result.status in (401, 403):
        raise restricted
    if result.status >= 500:
        raise FetchFailed("The store's website isn't responding right now. Try again later, or upload a photo.",
                          code="unreachable")
    if result.status != 200:
        return  # no robots.txt: everything is allowed
    parser = RobotFileParser()
    parser.parse(result.body.decode("utf-8", "replace").splitlines())
    token = user_agent.split("/", 1)[0]
    if not parser.can_fetch(token, page_url):
        raise restricted

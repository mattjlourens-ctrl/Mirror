import unittest

from backend.errors import AccessRestricted, FetchFailed
from backend.product_page import check_robots, find_product_images
from tests.backend_fakes import FakeWeb

PAGE = "https://shop.test/frames/aria?colour=black"


def urls(html, limit=10):
    return [ref.url for ref in find_product_images(html, PAGE, limit)]


class FindProductImagesTests(unittest.TestCase):
    def test_reads_json_ld_open_graph_twitter_link_and_img_in_trust_order(self):
        html = """
        <html><head>
          <meta property="og:image" content="/img/og.jpg">
          <meta name="twitter:image" content="https://cdn.shop.test/tw.jpg">
          <link rel="image_src" href="/img/link.jpg">
          <script type="application/ld+json">
            {"@context": "https://schema.org", "@graph": [
              {"@type": "Organization", "logo": "/brand.png"},
              {"@type": "Product", "name": "Aria", "image": ["/img/ld-1.jpg", {"@type": "ImageObject", "url": "/img/ld-2.jpg"}]}
            ]}
          </script>
        </head><body><img src="/img/gallery.jpg" alt="Aria glasses"></body></html>"""
        self.assertEqual(urls(html), [
            "https://shop.test/img/ld-1.jpg", "https://shop.test/img/ld-2.jpg", "https://shop.test/img/og.jpg",
            "https://cdn.shop.test/tw.jpg", "https://shop.test/img/link.jpg", "https://shop.test/img/gallery.jpg"])

    def test_skips_logos_icons_tiny_images_and_unsupported_formats(self):
        html = """
          <img src="/logo.png"><img src="/a.svg"><img src="/b.gif"><img src="data:image/png;base64,AAAA">
          <img src="/payment/visa.png"><img src="/x.jpg" alt="Store logo"><img src="/thumb.jpg" width="60" height="40">
          <img src="/photos/aria.jpg" alt="Aria">"""
        self.assertEqual(urls(html), ["https://shop.test/photos/aria.jpg"])

    def test_prefers_front_views_and_the_largest_srcset_entry(self):
        html = """
          <img src="/p/aria-side.jpg"><img src="/p/aria-front.jpg">
          <img srcset="/p/s.jpg 400w, /p/l.jpg 1600w, /p/m.jpg 800w" src="/p/s.jpg" alt="Aria angle">"""
        self.assertEqual(urls(html), ["https://shop.test/p/aria-front.jpg", "https://shop.test/p/aria-side.jpg",
                                      "https://shop.test/p/l.jpg"])

    def test_deduplicates_respects_base_href_and_limit(self):
        html = """<base href="https://cdn.shop.test/assets/">
          <meta property="og:image" content="a.jpg"><img src="a.jpg"><img src="b.jpg"><img src="c.jpg">"""
        self.assertEqual(urls(html, limit=2), ["https://cdn.shop.test/assets/a.jpg", "https://cdn.shop.test/assets/b.jpg"])

    def test_finds_image_urls_embedded_in_scripts_last(self):
        html = """<img src="/p/hero.jpg" alt="Aria">
          <script>self.__data = {"gallery": ["https:\\/\\/cdn.shop.test\\/a\\/front.jpg?w=728\\u0026h=728",
            "https://cdn.shop.test/b/side.webp", "https://cdn.shop.test/logo.png", "https://cdn.shop.test/x.svg"]}</script>"""
        self.assertEqual(urls(html), ["https://shop.test/p/hero.jpg", "https://cdn.shop.test/a/front.jpg?w=728&h=728",
                                      "https://cdn.shop.test/b/side.webp"])

    def test_survives_broken_markup_and_json(self):
        html = '<script type="application/ld+json">{not json</script><img src="/ok.jpg"<meta'
        self.assertIsInstance(urls(html), list)
        self.assertEqual(urls(""), [])


class RobotsTests(unittest.TestCase):
    def setUp(self):
        self.web = FakeWeb()
        self.fetcher = self.web.fetcher()
        self.addCleanup(self.fetcher.close)

    def test_missing_robots_allows(self):
        self.web.dns["shop.test"] = ["93.184.216.34"]
        check_robots(self.fetcher, PAGE, "MirrorBot/0.1")

    def test_disallow_rules_are_respected(self):
        self.web.add("https://shop.test/robots.txt", "User-agent: *\nDisallow: /frames/\n", content_type="text/plain")
        with self.assertRaises(AccessRestricted):
            check_robots(self.fetcher, PAGE, "MirrorBot/0.1")
        check_robots(self.fetcher, "https://shop.test/other", "MirrorBot/0.1")

    def test_rules_for_mirrorbot_apply(self):
        self.web.add("https://shop.test/robots.txt", "User-agent: MirrorBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
                     content_type="text/plain")
        with self.assertRaises(AccessRestricted):
            check_robots(self.fetcher, PAGE, "MirrorBot/0.1 (virtual try-on)")

    def test_forbidden_robots_means_no_access(self):
        self.web.add("https://shop.test/robots.txt", status=403)
        with self.assertRaises(AccessRestricted):
            check_robots(self.fetcher, PAGE, "MirrorBot/0.1")

    def test_server_errors_are_reported(self):
        self.web.add("https://shop.test/robots.txt", status=503)
        with self.assertRaises(FetchFailed):
            check_robots(self.fetcher, PAGE, "MirrorBot/0.1")


if __name__ == "__main__":
    unittest.main()

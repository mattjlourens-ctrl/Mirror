import unittest

from backend.errors import BlockedDestination, FetchFailed
from backend.safe_fetch import check_url, is_public_address
from tests.backend_fakes import PUBLIC_IP, FakeWeb


class AddressTests(unittest.TestCase):
    def test_private_and_special_addresses_are_not_public(self):
        for address in ["127.0.0.1", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254",
                        "0.0.0.0", "100.64.0.1", "224.0.0.1", "::1", "fc00::1", "fe80::1",
                        "::ffff:127.0.0.1", "::ffff:10.0.0.1", "2002:7f00:0001::1", "::"]:
            with self.subTest(address=address):
                self.assertFalse(is_public_address(address))

    def test_public_addresses_are_public(self):
        for address in [PUBLIC_IP, "8.8.8.8", "2606:4700:4700::1111"]:
            with self.subTest(address=address):
                self.assertTrue(is_public_address(address))


class UrlTests(unittest.TestCase):
    def test_rejects_unsupported_urls(self):
        for url in ["ftp://shop.test/a", "file:///etc/passwd", "javascript:alert(1)", "https://",
                    "https://user:pass@shop.test/", "http://shop.test:8080/", "https://shop.test:22/",
                    "https://shop.test/" + "a" * 3000, "http://[::1"]:
            with self.subTest(url=url):
                with self.assertRaises(FetchFailed):
                    check_url(url)

    def test_accepts_default_ports(self):
        for url in ["https://shop.test/p?id=1", "http://shop.test:80/", "https://shop.test:443/x"]:
            check_url(url)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.web = FakeWeb()
        self.fetcher = self.web.fetcher()
        self.addCleanup(self.fetcher.close)

    def test_connects_to_the_checked_address_with_the_real_host_and_sni(self):
        self.web.add("https://shop.test/frames", "<html></html>")
        result = self.fetcher.get("https://shop.test/frames", max_bytes=1000)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.url, "https://shop.test/frames")
        host, connected_to, path, extensions = self.web.requests[-1]
        self.assertEqual((host, connected_to, path), ("shop.test", PUBLIC_IP, "/frames"))
        self.assertEqual(extensions.get("sni_hostname"), "shop.test")

    def test_blocks_hosts_that_resolve_to_private_addresses(self):
        for ips in (["127.0.0.1"], ["10.1.2.3"], ["169.254.169.254"], [PUBLIC_IP, "192.168.0.10"], ["::1"]):
            with self.subTest(ips=ips):
                self.web.dns["internal.test"] = ips
                with self.assertRaises(BlockedDestination):
                    self.fetcher.get("https://internal.test/", max_bytes=1000)
        self.assertEqual(self.web.requests, [])

    def test_blocks_private_ip_literals(self):
        for url in ["http://127.0.0.1/", "http://[::1]/", "http://169.254.169.254/latest/meta-data/"]:
            with self.subTest(url=url):
                with self.assertRaises(BlockedDestination):
                    self.fetcher.get(url, max_bytes=1000)

    def test_follows_safe_redirects(self):
        self.web.add("http://shop.test/old", status=301, headers={"location": "https://shop.test/new"})
        self.web.add("https://shop.test/new", "ok")
        result = self.fetcher.get("http://shop.test/old", max_bytes=1000)
        self.assertEqual((result.url, result.body), ("https://shop.test/new", b"ok"))

    def test_blocks_redirects_to_internal_hosts(self):
        self.web.add("https://shop.test/go", status=302, headers={"location": "http://169.254.169.254/latest"})
        with self.assertRaises(BlockedDestination):
            self.fetcher.get("https://shop.test/go", max_bytes=1000)
        self.web.dns["evil.test"] = ["10.0.0.1"]
        self.web.add("https://shop.test/go2", status=307, headers={"location": "https://evil.test/"})
        with self.assertRaises(BlockedDestination):
            self.fetcher.get("https://shop.test/go2", max_bytes=1000)
        self.web.add("https://shop.test/go3", status=302, headers={"location": "file:///etc/passwd"})
        with self.assertRaises(FetchFailed):
            self.fetcher.get("https://shop.test/go3", max_bytes=1000)
        self.assertNotIn("evil.test", [host for host, *_ in self.web.requests])

    def test_limits_redirect_chains(self):
        self.web.add("https://shop.test/loop", status=302, headers={"location": "/loop"})
        with self.assertRaises(FetchFailed) as caught:
            self.fetcher.get("https://shop.test/loop", max_bytes=1000)
        self.assertEqual(caught.exception.code, "too_many_redirects")

    def test_rejects_oversized_bodies(self):
        self.web.add("https://shop.test/big", b"x" * 5000)
        with self.assertRaises(FetchFailed) as caught:
            self.fetcher.get("https://shop.test/big", max_bytes=4999)
        self.assertEqual(caught.exception.code, "too_large")
        self.assertEqual(len(self.fetcher.get("https://shop.test/big", max_bytes=5000).body), 5000)

    def test_unknown_hosts_fail_cleanly(self):
        with self.assertRaises(FetchFailed) as caught:
            self.fetcher.get("https://nowhere.test/", max_bytes=1000)
        self.assertEqual(caught.exception.code, "unknown_host")


if __name__ == "__main__":
    unittest.main()

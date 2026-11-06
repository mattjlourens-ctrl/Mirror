"""The website's copy of the try-on geometry must match the Python modules."""
import unittest

from tools.export_web_ar_data import outputs


class WebArDataTests(unittest.TestCase):
    def test_generated_typescript_is_up_to_date(self):
        for path, expected in outputs().items():
            with self.subTest(path=path.name):
                self.assertTrue(path.exists(), f"{path} is missing; run tools/export_web_ar_data.py")
                self.assertEqual(path.read_text(), expected,
                                 f"{path.name} is stale; run uv run python tools/export_web_ar_data.py")


if __name__ == "__main__":
    unittest.main()

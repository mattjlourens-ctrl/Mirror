import tempfile
import unittest
from pathlib import Path

from backend.images import SourceImage
from backend.storage import TempStore

IMAGE = SourceImage(b"\x89PNG fake", "image/png", 100, 100)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TempStoreTests(unittest.TestCase):
    def setUp(self):
        parent = tempfile.TemporaryDirectory()
        self.addCleanup(parent.cleanup)
        self.clock = Clock()
        self.store = TempStore(parent.name, ttl_seconds=60, max_items=3, clock=self.clock)
        self.addCleanup(self.store.close)

    def test_stores_and_returns_files_by_kind(self):
        image_id = self.store.put_image(IMAGE)
        item = self.store.get("image", image_id)
        self.assertEqual(item.path.read_bytes(), IMAGE.data)
        self.assertEqual(item.media_type, "image/png")
        self.assertIsNone(self.store.get("model", image_id))
        self.assertIsNone(self.store.get("image", "0" * 32))

    def test_models_are_moved_in(self):
        with self.store.work_dir() as work:
            glb = work / "glasses.glb"
            glb.write_bytes(b"glTF")
            model_id = self.store.put_model(glb)
            self.assertFalse(glb.exists())
        self.assertEqual(self.store.get("model", model_id).path.read_bytes(), b"glTF")

    def test_files_expire(self):
        image_id = self.store.put_image(IMAGE)
        path = self.store.get("image", image_id).path
        self.clock.now = 61
        self.assertIsNone(self.store.get("image", image_id))
        self.assertFalse(path.exists())

    def test_capacity_removes_the_oldest(self):
        ids = []
        for step in range(5):
            self.clock.now = step
            ids.append(self.store.put_image(IMAGE))
        self.assertIsNone(self.store.get("image", ids[0]))
        self.assertIsNotNone(self.store.get("image", ids[-1]))
        self.assertEqual(len(list((self.store.root / "images").iterdir())), 3)

    def test_work_dirs_are_removed_even_after_errors(self):
        with self.assertRaises(RuntimeError):
            with self.store.work_dir() as work:
                (work / "partial.glb").write_bytes(b"x")
                raise RuntimeError
        self.assertEqual(list((self.store.root / "work").iterdir()), [])

    def test_close_removes_everything(self):
        self.store.put_image(IMAGE)
        root = self.store.root
        self.store.close()
        self.assertFalse(Path(root).exists())


if __name__ == "__main__":
    unittest.main()

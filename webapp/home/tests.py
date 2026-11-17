"""Tests for the CHARM web app.

The detection models are never loaded here: uploads are processed by a stub,
so the suite runs without PyTorch, Ultralytics or the model weights.
"""
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from home import detection
from home.models import Contact, FileUpload
from home.templatetags.file_filters import endswith

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
    b"\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class MediaTestCase(TestCase):
    """Runs each test against a throwaway MEDIA_ROOT."""

    def setUp(self):
        self.media_root = tempfile.mkdtemp()
        override = override_settings(MEDIA_ROOT=self.media_root)
        override.enable()
        self.addCleanup(override.disable)
        self.addCleanup(shutil.rmtree, self.media_root, ignore_errors=True)


class PageTests(TestCase):
    def test_static_pages_render(self):
        for name in ["home", "about", "live", "contact", "upload"]:
            with self.subTest(page=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)


class ContactFormTests(TestCase):
    def test_submission_is_saved(self):
        response = self.client.post(reverse("contact"), {
            "name": "Ayesha Khan",
            "email": "ayesha@example.com",
            "phone": "03001234567",
            "message": "Can CHARM work with our CCTV cameras?",
        })
        self.assertEqual(response.status_code, 200)
        saved = Contact.objects.get()
        self.assertEqual(saved.name, "Ayesha Khan")
        self.assertEqual(saved.message, "Can CHARM work with our CCTV cameras?")


class UploadTests(MediaTestCase):
    def fake_detection(self, input_path, output_dir):
        """Stands in for the model pipeline: writes an output file like the real one."""
        out = Path(output_dir) / f"{Path(input_path).stem}_classified.jpg"
        out.write_bytes(b"processed")
        return str(out)

    def upload(self, name="flock.png", content=PNG_BYTES):
        return self.client.post(reverse("upload"), {"file": SimpleUploadedFile(name, content)})

    def test_successful_upload_redirects_to_result(self):
        with mock.patch("home.views.run_detection_and_tracking", side_effect=self.fake_detection):
            response = self.upload()
        record = FileUpload.objects.get()
        self.assertRedirects(response, reverse("result", args=[record.id]))
        self.assertEqual(record.result, "processed/flock_classified.jpg")

    def test_result_page_links_to_files_that_exist(self):
        with mock.patch("home.views.run_detection_and_tracking", side_effect=self.fake_detection):
            self.upload()
        record = FileUpload.objects.get()
        context = self.client.get(reverse("result", args=[record.id])).context

        for key in ("original_url", "result_url"):
            with self.subTest(url=key):
                url = context[key]
                self.assertTrue(url.startswith("/media/"))
                on_disk = Path(self.media_root) / url.removeprefix("/media/")
                self.assertTrue(on_disk.exists(), f"{key} points to a missing file: {url}")

    @override_settings(MAX_UPLOAD_SIZE=10)
    def test_oversized_upload_is_rejected(self):
        with mock.patch("home.views.run_detection_and_tracking") as detect:
            response = self.upload()
        self.assertEqual(response.status_code, 400)
        detect.assert_not_called()

    def test_upload_without_a_file_is_invalid(self):
        response = self.client.post(reverse("upload"), {})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(FileUpload.objects.count(), 0)

    def test_detection_failure_returns_to_upload_page(self):
        with mock.patch("home.views.run_detection_and_tracking", side_effect=RuntimeError("boom")):
            response = self.upload()
        self.assertRedirects(response, reverse("upload"))
        self.assertEqual(FileUpload.objects.count(), 0)

    def test_unknown_result_id_redirects(self):
        self.assertRedirects(self.client.get(reverse("result", args=[999])), reverse("upload"))


class QualityGateTests(TestCase):
    """calculate_quality decides which crops are sharp and well-lit enough to classify."""

    def test_textured_well_lit_crop_passes(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(40, 210, size=(120, 120, 3), dtype=np.uint8)
        score, ok = detection.calculate_quality(frame, (10, 10, 110, 110))
        self.assertTrue(ok)
        self.assertGreater(score, 0)

    def test_flat_crop_fails(self):
        frame = np.full((120, 120, 3), 128, dtype=np.uint8)
        _, ok = detection.calculate_quality(frame, (10, 10, 110, 110))
        self.assertFalse(ok)

    def test_too_dark_crop_fails(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 30, size=(120, 120, 3), dtype=np.uint8)
        _, ok = detection.calculate_quality(frame, (10, 10, 110, 110))
        self.assertFalse(ok)

    def test_empty_crop_is_handled(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertEqual(detection.calculate_quality(frame, (20, 20, 20, 20)), (0.0, False))


class ModelLoadingTests(TestCase):
    def test_missing_weights_give_a_clear_error(self):
        missing = {k: Path("/nonexistent") / f"{k}.pt" for k in detection.MODEL_PATHS}
        with mock.patch.dict(detection.MODEL_PATHS, missing), \
                mock.patch.object(detection, "_models", None):
            with self.assertRaisesRegex(FileNotFoundError, "CHARM_MODEL_DIR"):
                detection.get_models()


class TemplateFilterTests(TestCase):
    def test_endswith_is_case_insensitive_and_safe(self):
        self.assertTrue(endswith("/media/clip.MP4", ".mp4"))
        self.assertFalse(endswith("/media/photo.png", ".jpg"))
        self.assertFalse(endswith(None, ".jpg"))

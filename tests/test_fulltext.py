import contextlib
import hashlib
import io
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from refmind.services import fulltext
from refmind.services.paper_discovery import PaperCandidate, PaperLocation


def pdf_bytes(text):
    document = fitz.open()
    document.new_page().insert_text((40, 60), text)
    data = document.tobytes()
    document.close()
    return data


class Response:
    def __init__(self, body=b"", status=200, headers=None):
        self.status, self.headers = status, headers or {"Content-Type": "application/pdf"}
        self.stream = io.BytesIO(body)
    def getheader(self, name):
        return self.headers.get(name)
    def read(self, size):
        return self.stream.read(size)
    def close(self):
        self.stream.close()


class FulltextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidate = PaperCandidate("doi:10.1234/wanted", "A verified scientific paper title", doi="10.1234/wanted",
                    locations=(PaperLocation("https://public.example/paper.pdf", is_oa=True, version="acceptedVersion"),))

    def opener(self, response):
        @contextlib.contextmanager
        def open_url(*args):
            yield response
        return open_url

    def test_doi_verified_pdf_is_atomic_and_hashed(self):
        body = pdf_bytes("A verified scientific paper title\ndoi:10.1234/wanted")
        with patch.object(fulltext, "_open_public", self.opener(Response(body))):
            result = fulltext.fetch_public_pdf(self.candidate, self.root)
        self.assertEqual(result.identity_status, "matched")
        self.assertEqual(result.sha256, hashlib.sha256(body).hexdigest())
        self.assertEqual(result.path.read_bytes(), body)
        self.assertEqual(list(self.root.glob("*.downloading")), [])

    def test_unknown_identity_requires_review(self):
        with patch.object(fulltext, "_open_public", self.opener(Response(pdf_bytes("Undetermined document")))):
            result = fulltext.fetch_public_pdf(self.candidate, self.root)
        self.assertEqual(result.identity_status, "unknown")
        self.assertTrue(result.warnings)

    def test_explicit_other_doi_is_rejected_and_removed(self):
        with patch.object(fulltext, "_open_public", self.opener(Response(pdf_bytes("doi:10.1234/other")))):
            with self.assertRaises(fulltext.FulltextIdentityError):
                fulltext.fetch_public_pdf(self.candidate, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_known_hash_mismatch_cannot_be_accepted_by_title(self):
        with patch.object(fulltext, "_open_public", self.opener(Response(pdf_bytes(self.candidate.title)))):
            with self.assertRaises(fulltext.FulltextIdentityError):
                fulltext.fetch_public_pdf(self.candidate, self.root, expected_sha256="0" * 64)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_html_and_bad_pdf_are_rejected(self):
        for response in (Response(b"<html>login</html>", headers={"Content-Type": "text/html"}), Response(b"%PDF-not-a-real-pdf")):
            with self.subTest(response=response), patch.object(fulltext, "_open_public", self.opener(response)):
                with self.assertRaises(fulltext.FulltextError):
                    fulltext.fetch_public_pdf(self.candidate, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_content_length_and_stream_size_are_both_bounded(self):
        for response in (Response(b"abc", headers={"Content-Length": "10000"}), Response(b"%PDF-" + b"x" * 200)):
            with self.subTest(response=response), patch.object(fulltext, "_open_public", self.opener(response)):
                with self.assertRaises(fulltext.FulltextError):
                    fulltext.fetch_public_pdf(self.candidate, self.root, max_bytes=100)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_non_public_addresses_and_unsafe_protocols_are_rejected(self):
        with patch.object(fulltext.socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]):
            for url in ("https://public.example/a.pdf", "file:///tmp/a.pdf", "https://u:p@public.example/a.pdf", "http://public.example:8080/a.pdf"):
                with self.subTest(url=url), self.assertRaises(fulltext.FulltextError):
                    fulltext._public_url(url)

    def test_every_redirect_is_revalidated_before_connecting(self):
        def resolve(host, *args, **kwargs):
            address = "127.0.0.1" if host == "private.example" else "8.8.8.8"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]
        visited = []
        @contextlib.contextmanager
        def open_url(url, timeout):
            fulltext._public_url(url)
            visited.append(url)
            yield Response(status=302, headers={"Location": "https://private.example/secret"})
        with patch.object(fulltext.socket, "getaddrinfo", side_effect=resolve), patch.object(fulltext, "_open_public", open_url):
            with self.assertRaises(fulltext.FulltextError):
                fulltext.fetch_public_pdf(self.candidate, self.root)
        self.assertEqual(visited, [self.candidate.locations[0].url])

    def test_public_ip_is_pinned_after_dns_validation(self):
        class Connection:
            sock = None
            def __init__(self, *args, **kwargs):
                pass
            def request(self, *args, **kwargs):
                self._create_connection(("attacker.example", 443), timeout=2)
            def getresponse(self):
                return Response()
            def close(self):
                pass
        with patch.object(fulltext.socket, "getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]), \
             patch.object(fulltext.http.client, "HTTPSConnection", Connection), patch.object(fulltext.socket, "create_connection") as connect:
            with fulltext._open_public("https://attacker.example/paper.pdf", 2):
                pass
        self.assertEqual(connect.call_args.args[0], ("8.8.8.8", 443))

    def test_candidate_replay_does_not_bypass_network_validation(self):
        data = self.candidate.to_dict()
        data["locations"][0]["url"] = "http://127.0.0.1/secret.pdf"
        candidate = PaperCandidate.from_dict(data)
        with self.assertRaises(fulltext.FulltextError):
            fulltext.fetch_public_pdf(candidate, self.root)

    def test_non_oa_candidate_is_not_fetched(self):
        candidate = PaperCandidate("x", "Paper", locations=(PaperLocation("https://public.example/p.pdf", is_oa=None),))
        with patch.object(fulltext, "_open_public") as request, self.assertRaises(fulltext.FulltextError):
            fulltext.fetch_public_pdf(candidate, self.root)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

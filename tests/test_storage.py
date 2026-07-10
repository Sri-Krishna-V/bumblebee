"""LocalStorage, path normalization, and scheme dispatch characterization."""

import pytest

from bumblebee.storage import FsspecStorage, LocalStorage, normalize_relative_path, resolve_storage


def test_normalize_relative_path():
    assert normalize_relative_path("a/b/c.pdf") == "a/b/c.pdf"
    assert normalize_relative_path("/a//b/") == "a/b"
    assert normalize_relative_path("./a\\b") == "a/b"
    assert normalize_relative_path("") == ""
    assert normalize_relative_path(".") == ""


def test_local_storage_round_trip(tmp_path):
    storage = LocalStorage()
    uri = str(tmp_path / "sub" / "file.json")
    storage.write_json(uri, {"x": 1})
    assert storage.exists(uri)
    assert storage.read_json(uri) == {"x": 1}

    text_uri = storage.join(str(tmp_path), "notes/readme.md")
    storage.write_text(text_uri, "hello")
    assert storage.read_bytes(text_uri) == b"hello"


def test_local_storage_list_pdfs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.pdf").write_bytes(b"%PDF")
    (tmp_path / "two.pdf").write_bytes(b"%PDF")
    (tmp_path / "ignored.txt").write_text("no")

    documents = LocalStorage().list_pdfs(str(tmp_path))
    assert [d.relative_path for d in documents] == ["a/one.pdf", "two.pdf"]
    assert all(d.metadata["size_bytes"] == 4 for d in documents)
    assert len({d.input_id for d in documents}) == 2

    single = LocalStorage().list_pdfs(str(tmp_path / "two.pdf"))
    assert [d.relative_path for d in single] == ["two.pdf"]


def test_local_storage_list_files(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x.txt").write_text("x")
    (tmp_path / "y.txt").write_text("y")
    files = LocalStorage().list_files(str(tmp_path))
    assert files == sorted([str(tmp_path / "d" / "x.txt"), str(tmp_path / "y.txt")])


def test_resolve_storage_scheme_dispatch(tmp_path):
    assert isinstance(resolve_storage(str(tmp_path)), LocalStorage)
    assert isinstance(resolve_storage(f"file://{tmp_path}"), LocalStorage)
    # Cloud backends construct without their optional deps (fsspec loads lazily on use).
    assert isinstance(resolve_storage("s3://bucket/key"), FsspecStorage)
    assert isinstance(resolve_storage("gs://bucket/key"), FsspecStorage)
    assert isinstance(resolve_storage("az://container/key"), FsspecStorage)
    with pytest.raises(ValueError):
        resolve_storage("ftp://nope")


def test_s3_options_from_env(monkeypatch):
    from bumblebee.storage import _options_for_scheme

    monkeypatch.setenv("BUMBLEBEE_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("BUMBLEBEE_S3_REGION", "eu-central-1")
    options = _options_for_scheme("s3")
    assert options == {"client_kwargs": {"endpoint_url": "http://minio:9000", "region_name": "eu-central-1"}}


def test_azure_options_from_sas_url(monkeypatch):
    from bumblebee.storage import _options_for_scheme

    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("BUMBLEBEE_AZURE_SAS_URL", "https://acct.blob.core.windows.net/container?sig=abc")
    options = _options_for_scheme("az")
    assert options["account_name"] == "acct"
    assert options["sas_token"] == "sig=abc"

"""Release smoke test for built bumblebee distributions."""

import os
from importlib.metadata import metadata, version as installed_version

import bumblebee
from bumblebee import BatchPolicy, OcrConfig


def assert_equal(actual: object, expected: object, message: str) -> None:
    """Raise a useful smoke-test failure when values differ."""
    if actual != expected:
        raise RuntimeError(f"{message}: expected {expected!r}, got {actual!r}")


def main() -> None:
    """Exercise the installed package without importing GPU-only modules."""
    expected_version = os.getenv("EXPECTED_PACKAGE_VERSION")
    if expected_version:
        assert_equal(
            installed_version("bumblebee"),
            expected_version,
            "installed package version does not match the release tag",
        )
        assert_equal(bumblebee.__version__, expected_version, "bumblebee.__version__ mismatch")

    package_metadata = metadata("bumblebee")
    assert_equal(package_metadata["Name"], "bumblebee", "unexpected package name")
    assert_equal(package_metadata["License-Expression"], "Apache-2.0", "unexpected package license")

    config = OcrConfig()
    assert_equal(config.pdf_dpi, 100, "unexpected default OCR DPI")
    assert_equal(BatchPolicy().max_docs, 64, "unexpected default batch size")

    print("Release smoke test succeeded")


if __name__ == "__main__":
    main()

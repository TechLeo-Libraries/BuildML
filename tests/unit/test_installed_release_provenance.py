"""Keep old-pip provenance checks strict without rejecting valid wheel installs."""
import pytest

from scripts.check_installed_release import archive_sha256


@pytest.mark.parametrize("archive,expected", [
    ({"hashes": {"sha256": "abc"}}, "abc"),
    ({"hash": "sha256=abc"}, "abc"),
    ({"hash": "md5=abc"}, None),
    ({"hash": "abc"}, None),
    ({}, None),
    ({"hashes": {}, "hash": "sha256=abc"}, None),
    ({"hashes": {"sha256": "new"}, "hash": "sha256=old"}, "new"),
])
def test_pep610_archive_hash_formats(archive, expected):
    assert archive_sha256({"archive_info": archive}) == expected

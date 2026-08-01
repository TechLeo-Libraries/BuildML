from pathlib import Path

import pytest

from buildml.reporting import (
    ReportSection,
    encode_asset,
    render_card,
    render_report,
    render_table,
    write_report,
)


def test_components_escape_untrusted_content() -> None:
    table = render_table(
        [{"<column>": "<script>alert('cell')</script>"}],
        caption='Results "and" risks',
    )
    card = render_card("<img src=x onerror=alert(1)>", "A & B")

    assert "<script>" not in table
    assert "&lt;script&gt;" in table
    assert "&lt;column&gt;" in table
    assert 'scope="col"' in table
    assert "<img" not in card
    assert "&lt;img" in card
    assert "A &amp; B" in card


def test_report_is_offline_and_has_accessibility_landmarks(tmp_path: Path) -> None:
    report = render_report(
        "Risk <review>",
        [
            ReportSection(
                key="overview",
                title="Overview & scope",
                summary="What this report covers.",
                body_html=render_table([{"metric": "rows", "value": 12}], caption="Dataset size"),
            ),
            ReportSection(
                key="next-steps",
                title="Next steps",
                body_html=render_card("Review", "Confirm the target role.", tone="warn"),
            ),
        ],
        subtitle="A local artifact",
        metadata={"partition": "train", "rows": 12},
    )

    assert report.startswith("<!doctype html>")
    assert '<html lang="en">' in report
    assert '<meta name="viewport"' in report
    assert 'href="#main-content"' in report
    assert '<nav class="bml-nav" aria-label="Report sections">' in report
    assert '<main id="main-content" tabindex="-1">' in report
    assert 'aria-pressed="false"' in report
    assert "Risk &lt;review&gt;" in report
    assert "http://" not in report
    assert "https://" not in report
    assert "<link " not in report
    assert "<style>" in report and "<script>" in report

    destination = write_report(
        tmp_path / "nested" / "report.html",
        "BuildML report",
        [ReportSection("summary", "Summary", "<p>Trusted component HTML.</p>")],
    )
    assert destination.exists()
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_asset_encoding_uses_data_uri(tmp_path: Path) -> None:
    image = tmp_path / "dot.png"
    image.write_bytes(b"\x89PNG\r\n")
    encoded = encode_asset(image)
    assert encoded.startswith("data:image/png;base64,")
    assert encode_asset(b"plain", media_type="text/plain") == "data:text/plain;base64,cGxhaW4="


def test_report_rejects_empty_or_duplicate_sections() -> None:
    with pytest.raises(ValueError, match="at least one"):
        render_report("Empty", [])
    with pytest.raises(ValueError, match="Duplicate"):
        render_report(
            "Duplicate",
            [
                ReportSection("same key", "One", "<p>One</p>"),
                ReportSection("same-key", "Two", "<p>Two</p>"),
            ],
        )


"""Report budgets and real figure assets survive portable research exports."""

import base64
import binascii
import struct
import zlib
from copy import deepcopy
from html.parser import HTMLParser

import numpy as np
import pytest

from buildml.eda.html_report import export_eda_html
from buildml.eda.report import EDAReport


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.images = []
        self.text = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs))

    def handle_data(self, data):
        self.text.append(data)


def png(pixel):
    """Encode one actual RGB pixel without an image-library dependency."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes([0, *pixel])))
        + chunk(b"IEND", b"")
    )


def image_bytes(image):
    prefix, encoded = image["src"].split(",", 1)
    assert prefix == "data:image/png;base64"
    result = base64.b64decode(encoded, validate=True)
    assert result.startswith(b"\x89PNG\r\n\x1a\n")
    return result


def payload():
    # A deliberately partial saved report: measurements that were not run remain absent.
    return {
        "overview": {"n_rows": 4, "n_columns": 1, "memory_bytes_approx": 32},
        "quality": {},
        "findings": [],
        "recommendation_details": [],
        "warnings": [],
    }


def report_for(data, figures=None):
    return EDAReport(
        overview=data.get("overview", {}),
        quality=data.get("quality", {}),
        univariate={},
        bivariate={},
        multivariate={},
        target={},
        outliers={},
        figures={} if figures is None else figures,
        figure_paths=data.get("figure_paths", {}),
    )


BAD = [
    -1,
    np.int64(-1),
    True,
    False,
    np.bool_(True),
    np.bool_(False),
    None,
    "2",
    1.5,
    np.float64(2),
    float("nan"),
    float("inf"),
    float("-inf"),
    1 + 0j,
    [1],
    np.array([1, 2]),
    {},
]


@pytest.mark.parametrize("field", ["max_figures", "max_html_bytes"])
@pytest.mark.parametrize("value", BAD, ids=lambda v: repr(v))
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_budget_is_refused_before_creating_or_replacing_any_output(
    field, value, existing, tmp_path
):
    folder = tmp_path / "destination"
    target = folder / "report.html"
    if existing:
        folder.mkdir()
        target.write_bytes(b"previous complete report\x00")
    before = deepcopy(payload())
    with pytest.raises(ValueError, match=field):
        export_eda_html(before, target, **{field: value})
    assert before == payload()
    if existing:
        assert target.read_bytes() == b"previous complete report\x00"
        assert list(folder.iterdir()) == [target]
    else:
        assert not folder.exists()


@pytest.mark.parametrize("value", [0, np.int64(0)])
def test_zero_byte_budget_is_refused_before_creating_parent(value, tmp_path):
    target = tmp_path / "destination" / "report.html"
    with pytest.raises(ValueError, match="max_html_bytes"):
        export_eda_html(payload(), target, max_html_bytes=value)
    assert not target.parent.exists()


@pytest.mark.parametrize("budget", [0, np.int64(1), np.uint64(2), 2**80])
def test_valid_integer_figure_budget_counts_only_embedded_paths(budget, tmp_path):
    data = payload()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    paths = {}
    for i, color in enumerate(colors):
        path = tmp_path / f"image-{i}.png"
        path.write_bytes(png(color))
        paths[f"figure-{i}"] = str(path)
    data["figure_paths"] = paths
    target = export_eda_html(
        data, tmp_path / "report.html", max_figures=budget, max_html_bytes=np.uint64(2**64 - 1)
    )
    doc = Document(target.read_text(encoding="utf-8"))
    count = min(3, int(budget))
    assert [im["alt"] for im in doc.images] == [f"figure-{i}" for i in range(count)]
    assert [image_bytes(im) for im in doc.images] == [png(color) for color in colors[:count]]
    for name in list(paths)[count:]:
        assert f"{name}: figure budget of {int(budget)} reached" in "".join(doc.text)


@pytest.mark.parametrize("boundary", ["direct", "report"])
@pytest.mark.parametrize("name", ["ordinary", 'quoted"<&>', "Unicode 漢字\nline"])
def test_path_images_embed_exact_valid_bytes_and_survive_source_removal(boundary, name, tmp_path):
    raw = png((12, 34, 56))
    path = tmp_path / "picture.png"
    path.write_bytes(raw)
    data = payload()
    data["figure_paths"] = {name: str(path)}
    if boundary == "report":
        result = report_for(data).save_html(tmp_path / "report.html", html_format="research")
    else:
        result = export_eda_html(data, tmp_path / "report.html")
    path.unlink()
    doc = Document(result.read_text(encoding="utf-8"))
    assert len(doc.images) == 1 and doc.images[0]["alt"] == name
    assert image_bytes(doc.images[0]) == raw
    assert "Fig. M.1 - " + name in "".join(doc.text)


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_failed_path_and_analysis_entries_do_not_consume_successful_figure_budget(budget, tmp_path):
    missing = tmp_path / "missing-&.png"
    good = tmp_path / "good.png"
    good.write_bytes(png((9, 8, 7)))
    data = payload()
    data["figure_paths"] = {"missing": str(missing), "good": str(good)}
    target = export_eda_html(
        data,
        tmp_path / "report.html",
        figures={"unavailable": {"error": "native analyzer <unavailable>"}},
        max_figures=budget,
    )
    doc = Document(target.read_text(encoding="utf-8"))
    text = "".join(doc.text)
    assert [im["alt"] for im in doc.images] == ([] if budget == 0 else ["good"])
    assert "Skipped plots" in text
    if budget:
        assert "unavailable: native analyzer <unavailable>" in text
        assert "missing:" in text and missing.name in text
        assert image_bytes(doc.images[0]) == good.read_bytes()


@pytest.mark.parametrize("boundary", ["direct", "report"])
def test_native_matplotlib_figure_takes_precedence_over_same_name_path(boundary, tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1, 1))
    axis = figure.subplots()
    axis.plot([0, 1], [0, 1])
    path = tmp_path / "fallback.png"
    path.write_bytes(png((1, 2, 3)))
    data = payload()
    data["figure_paths"] = {"same": str(path)}
    if boundary == "report":
        target = report_for(data, {"same": figure}).save_html(
            tmp_path / "report.html", html_format="research"
        )
    else:
        target = export_eda_html(data, tmp_path / "report.html", figures={"same": figure})
    doc = Document(target.read_text(encoding="utf-8"))
    assert len(doc.images) == 1 and doc.images[0]["alt"] == "same"
    raw = image_bytes(doc.images[0])
    assert raw != path.read_bytes()
    width, height = struct.unpack(">II", raw[16:24])
    assert width > 20 and height > 20
    assert "same: figure budget" not in "".join(doc.text)


def test_real_detached_matplotlib_canvas_uses_retained_path_and_discloses_failure(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1, 1))
    figure.set_canvas(None)
    path = tmp_path / "retained.png"
    path.write_bytes(png((77, 88, 99)))
    data = payload()
    data["figure_paths"] = {"recoverable": str(path)}
    target = export_eda_html(
        data, tmp_path / "report.html", figures={"recoverable": figure}, max_figures=1
    )
    doc = Document(target.read_text(encoding="utf-8"))
    assert len(doc.images) == 1 and image_bytes(doc.images[0]) == path.read_bytes()
    text = "".join(doc.text)
    assert "recoverable:" in text and "Skipped plots" in text


@pytest.mark.parametrize("title", ["ASCII", "漢字🙂" * 9])
def test_final_byte_limit_is_inclusive_and_preserves_existing_file_on_failure(title, tmp_path):
    data = payload()
    target = tmp_path / "unbounded.html"
    export_eda_html(data, target, title=title, include_raw_appendix=False)
    raw = target.read_bytes()
    size = target.stat().st_size
    assert len(raw) == size
    exact = tmp_path / "exact.html"
    export_eda_html(
        data, exact, title=title, include_raw_appendix=False, max_html_bytes=np.int64(size)
    )
    assert exact.read_bytes() == raw
    blocked = tmp_path / "blocked.html"
    blocked.write_bytes(b"previous file")
    with pytest.raises(ValueError, match="Report size"):
        export_eda_html(
            data, blocked, title=title, include_raw_appendix=False, max_html_bytes=size - 1
        )
    assert blocked.read_bytes() == b"previous file"
    if title != "ASCII":
        assert len(raw.decode("utf-8")) < size
        with pytest.raises(ValueError, match="Report size"):
            export_eda_html(
                data,
                tmp_path / "characters.html",
                title=title,
                include_raw_appendix=False,
                max_html_bytes=len(raw.decode("utf-8")),
            )


def test_raw_appendix_is_omitted_before_rejecting_large_export_without_losing_image(tmp_path):
    data = payload()
    data["raw_consumer_marker"] = "appendix-only-marker" + ("漢字" * 60000)
    path = tmp_path / "image.png"
    path.write_bytes(png((123, 45, 67)))
    data["figure_paths"] = {"retained image": str(path)}
    large = export_eda_html(data, tmp_path / "large.html")
    compact = export_eda_html(data, tmp_path / "compact.html", include_raw_appendix=False)
    limit = compact.stat().st_size + 4096
    assert large.stat().st_size > limit
    bounded = export_eda_html(data, tmp_path / "bounded.html", max_html_bytes=limit)
    doc = Document(bounded.read_text(encoding="utf-8"))
    text = "".join(doc.text)
    assert bounded.stat().st_size <= limit and "Raw appendix omitted by output budget" in text
    assert "appendix-only-marker" not in text
    assert len(doc.images) == 1 and image_bytes(doc.images[0]) == path.read_bytes()

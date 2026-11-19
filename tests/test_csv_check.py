from pathlib import Path

import pytest

from anveshak.postprocess.csv_check import HEADER, CSVValidationError, assert_csv_valid, check_csv

SIZES = {"t1": (100, 80), "t2": (100, 80)}
CLASSES = ["AAA", "BBB"]
H = ",".join(HEADER)


def write(tmp_path: Path, body: str, header: str = H, raw: bytes | None = None) -> Path:
    p = tmp_path / "pred.csv"
    p.write_bytes(raw if raw is not None else (header + "\n" + body).encode("utf-8"))
    return p


def errs(tmp_path, body, **kw):
    return check_csv(write(tmp_path, body, **kw), CLASSES, SIZES).errors


def test_valid_file_passes(tmp_path):
    body = "t1,AAA,0.9,1,2,50,60\nt1,unknown,0.5,10.5,10.5,20.25,30\nt2,BBB,1.0,0,0,100,80\n"
    res = check_csv(write(tmp_path, body), CLASSES, SIZES)
    assert res.ok and res.n_rows == 3 and res.n_images_with_rows == 2
    assert_csv_valid(write(tmp_path, body), CLASSES, SIZES)


def test_header_must_be_exact(tmp_path):
    assert any("header" in e for e in errs(tmp_path, "t1,AAA,0.9,1,2,50,60\n", header="image_id,class,score,xmin,ymin,xmax,ymax"))


def test_class_case_sensitive(tmp_path):
    e = errs(tmp_path, "t1,aaa,0.9,1,2,50,60\n")
    assert any("did you mean 'AAA'" in x for x in e)
    assert any("Unknown" in x or "UNKNOWN" in x or "not a valid" in x for x in errs(tmp_path, "t1,Unknown,0.9,1,2,50,60\n"))
    assert errs(tmp_path, "t1,ZZZ,0.9,1,2,50,60\n")


@pytest.mark.parametrize("score", ["1.0001", "-0.1", "nan", "inf", "abc"])
def test_bad_scores(tmp_path, score):
    assert errs(tmp_path, f"t1,AAA,{score},1,2,50,60\n")


@pytest.mark.parametrize(
    "box",
    ["10,2,10,60", "10,2,5,60", "1,20,50,20", "1,20,50,10", "-1,2,50,60", "1,-2,50,60", "1,2,101,60", "1,2,50,81", "a,2,50,60", "nan,2,50,60"],
)
def test_bad_boxes(tmp_path, box):
    assert errs(tmp_path, f"t1,AAA,0.5,{box}\n")


def test_boundary_box_ok(tmp_path):
    assert not errs(tmp_path, "t1,AAA,0.5,0,0,100,80\n")


def test_duplicates_detected(tmp_path):
    assert any("duplicate" in e for e in errs(tmp_path, "t1,AAA,0.5,1,2,50,60\nt1,AAA,0.5,1,2,50,60\n"))
    assert not errs(tmp_path, "t1,AAA,0.5,1,2,50,60\nt1,AAA,0.6,1,2,50,60\n")  # different score = different row


def test_max_rows_per_image(tmp_path):
    body = "".join(f"t1,AAA,0.5,{i % 40},{i // 40},{50 + i % 40},{60 + i // 40}\n" for i in range(1001))
    assert any("1001 rows" in e for e in errs(tmp_path, body))
    assert not any("rows" in e for e in errs(tmp_path, "".join(f"t1,AAA,0.5,{i % 40},{i // 40},{50 + i % 40},{60 + i // 40}\n" for i in range(1000))))


def test_unknown_image_id_and_wrong_columns(tmp_path):
    assert any("not a known test image" in e for e in errs(tmp_path, "zzz,AAA,0.5,1,2,50,60\n"))
    assert any("7 columns" in e for e in errs(tmp_path, "t1,AAA,0.5,1,2,50\n"))


def test_bom_and_non_utf8(tmp_path):
    raw = b"\xef\xbb\xbf" + (H + "\nt1,AAA,0.5,1,2,50,60\n").encode()
    assert any("BOM" in e for e in check_csv(write(tmp_path, "", raw=raw), CLASSES, SIZES).errors)
    bad = write(tmp_path, "", raw=(H + "\nt1,A\xe9,0.5,1,2,50,60\n").encode("latin-1"))
    assert not check_csv(bad, CLASSES, SIZES).ok


def test_missing_images_are_warned_and_assert_raises(tmp_path):
    res = check_csv(write(tmp_path, "t1,AAA,0.5,1,2,50,60\n"), CLASSES, SIZES)
    assert res.ok and res.images_without_rows == ["t2"] and res.warnings
    with pytest.raises(CSVValidationError):
        assert_csv_valid(write(tmp_path, "t1,AAA,7,1,2,50,60\n"), CLASSES, SIZES)

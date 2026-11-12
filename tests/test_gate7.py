from __future__ import annotations

import hashlib

from state_alpha.reporting.gate7 import FIGURE_FILES, SCORE_ORDER, sha256_file


def test_gate7_contract_has_ten_scores_and_seven_figures() -> None:
    assert len(SCORE_ORDER) == 10
    assert len(set(SCORE_ORDER)) == 10
    assert len(FIGURE_FILES) == 7
    assert len(set(FIGURE_FILES)) == 7


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    payload = b"frozen presentation evidence\n"
    path = tmp_path / "evidence.txt"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

"""Check that disease/target names render as Open Targets hyperlinks.

Seam: _render_results_table(results, columns, link_columns) -> HTML string.
Expected values are literal URLs written out here (the Open Targets platform
URL scheme), not recomputed by the code under test.

Run: python3 test_render_links.py
"""
import re

import pandas as pd

from app import _render_results_table


def test_gene_name_links_to_target_page():
    results = pd.DataFrame({
        "Rank": [1],
        "Gene": ["SCN1A"],
        "Gene Name": ["sodium voltage-gated channel alpha subunit 1"],
        "Open Targets link": ["https://platform.opentargets.org/target/ENSG00000144285"],
    })
    out = _render_results_table(results, ["Rank", "Gene", "Gene Name"], {"Gene": "Open Targets link"})
    assert 'href="https://platform.opentargets.org/target/ENSG00000144285"' in out, out
    assert ">SCN1A</a>" in out, "gene symbol should be the link text"
    assert 'target="_blank"' in out and 'rel="noopener noreferrer"' in out
    # The URL itself must not leak into a visible cell (column was dropped).
    assert out.count("platform.opentargets.org") == 1, "URL should appear once, as the href only"
    print("PASS: gene symbol links to its Open Targets target page")


def test_disease_name_links_including_mondo():
    results = pd.DataFrame({
        "Rank": [1, 2],
        "Disease": ["epilepsy", "Crohn disease"],
        "Disease ID": ["EFO_0000474", "MONDO_0005011"],
        "Open Targets link": [
            "https://platform.opentargets.org/disease/EFO_0000474",
            "https://platform.opentargets.org/disease/MONDO_0005011",
        ],
    })
    out = _render_results_table(results, ["Rank", "Disease", "Disease ID"], {"Disease": "Open Targets link"})
    assert 'href="https://platform.opentargets.org/disease/EFO_0000474"' in out
    assert 'href="https://platform.opentargets.org/disease/MONDO_0005011"' in out, "MONDO ids must link too"
    assert ">epilepsy</a>" in out and ">Crohn disease</a>" in out
    print("PASS: disease names link to their Open Targets pages (EFO and MONDO)")


def test_missing_or_unsafe_url_falls_back_to_plain_text():
    results = pd.DataFrame({
        "Gene": ["ABC1", "DEF2", "GHI3"],
        "Open Targets link": ["", None, "javascript:alert(1)"],
    })
    out = _render_results_table(results, ["Gene"], {"Gene": "Open Targets link"})
    assert "<a href" not in out, "must not emit a link for empty/None/unsafe URLs"
    assert "javascript:" not in out, "unsafe scheme must never reach the output"
    for sym in ("ABC1", "DEF2", "GHI3"):
        assert sym in out
    print("PASS: empty/None/unsafe URLs fall back to plain text (no javascript: injection)")


def test_html_in_name_is_escaped():
    results = pd.DataFrame({
        "Disease": ["<script>alert(1)</script>"],
        "Open Targets link": ["https://platform.opentargets.org/disease/EFO_0000001"],
    })
    out = _render_results_table(results, ["Disease"], {"Disease": "Open Targets link"})
    assert "<script>" not in out, "link text must be HTML-escaped"
    assert "&lt;script&gt;" in out
    print("PASS: link text is HTML-escaped")





def test_prepare_display_frame_keeps_url_column():
    """Integration gap that shipped: DISPLAY_COLUMNS subsetting stripped the
    URL column before rendering, so live tables had zero links while the
    renderer-level tests above stayed green. Exercise the real preparer."""
    import numpy as np
    import pandas as pd
    from app import _prepare_display_frame, DISPLAY_COLUMNS, _render_results_table

    results = pd.DataFrame({
        "rank": [1], "approvedSymbol": ["TNF"], "approvedName": ["tumor necrosis factor"],
        "otrec_score": [0.9], "ot_score": [0.5], "ottree_pred": [0.8],
        "tractability": [["SM Approved Drug"]], "known_label": [1.0],
        "functionDescriptions": ["cytokine"], "targetId": ["ENSG00000232810"],
    })
    display_df = _prepare_display_frame(results)
    assert "Open Targets link" in display_df.columns, "URL column stripped by preparer"
    out = _render_results_table(display_df, DISPLAY_COLUMNS, {"Gene": "Open Targets link"})
    assert 'href="https://platform.opentargets.org/target/ENSG00000232810"' in out
    assert ">TNF</a>" in out
    print("PASS: full display-frame path renders the Gene hyperlink")


if __name__ == "__main__":
    test_gene_name_links_to_target_page()
    test_prepare_display_frame_keeps_url_column()
    test_disease_name_links_including_mondo()
    test_missing_or_unsafe_url_falls_back_to_plain_text()
    test_html_in_name_is_escaped()
    print("PASS: Open Targets link rendering verified")

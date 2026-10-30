from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any
from urllib.parse import quote

import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st


APP_TITLE = "Cross-Polarity Network Enricher"
APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
LOGO_PATH = APP_DIR / "assets" / "cpne_logo.png"
HOWTO_PATH = APP_DIR / "HOW_TO_USE.md"
DEMO_POSITIVE_TASK_ID = "4a83b08f0c9c41888b7db29c267f3cf1"
DEMO_NEGATIVE_TASK_ID = "0bd4d95478e7499dad2ffead34ddec13"
DEMO_TASK_CACHE_PATH = EXAMPLES_DIR / "gnps2_demo_task_cache.zip"
PROTON = 1.007276466621
WATER = 18.010564684

# Each entry is: display label -> (total ion mass shift relative to neutral M, |charge|).
POSITIVE_ADDUCTS: dict[str, tuple[float, int]] = {
    "[M+H]+": (PROTON, 1),
    "[M+Na]+": (22.989218, 1),
    "[M+K]+": (38.963158, 1),
    "[M+NH4]+": (18.033823, 1),
    "[M+2H]2+": (2 * PROTON, 2),
    "[M+H+Na]2+": (PROTON + 22.989218, 2),
}

NEGATIVE_ADDUCTS: dict[str, tuple[float, int]] = {
    "[M-H]-": (-PROTON, 1),
    "[M-H2O-H]-": (-PROTON - WATER, 1),
    "[M+Cl]-": (34.969402, 1),
    "[M+FA-H]-": (44.998201, 1),
    "[M+Ac-H]-": (59.013851, 1),
    "[M+Na-2H]-": (20.974665, 1),
    "[M-2H]2-": (-2 * PROTON, 2),
}

MZ_ALIASES = [
    "row m/z", "parent mass", "precursor m/z", "precursor mass", "specmz",
    "mz", "m/z", "mass", "premass", "parentmass", "observed mz",
]
RT_ALIASES = [
    "row retention time", "rtmean", "retention time", "retention_time",
    "rt (min)", "rt", "retentiontime", "retime", "ret time",
]
ID_ALIASES = [
    "row id", "cluster index", "clusteridx", "feature id", "feature_id",
    "#clusteridx", "#scan#", "scan", "scan number", "id", "feature", "node id",
]


def _norm_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _clean_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    # Make IDs exported as 123.0 equivalent to 123, without altering nonnumeric IDs.
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_norm_name(c): c for c in columns}
    for alias in aliases:
        match = normalized.get(_norm_name(alias))
        if match is not None:
            return match
    # Soft fallbacks for common column naming variants.
    if aliases is MZ_ALIASES:
        for c in columns:
            n = _norm_name(c)
            if n in {"mz", "rowmz", "parentmass", "precursormz", "specmz"} or ("precursor" in n and "mass" in n):
                return c
    if aliases is RT_ALIASES:
        for c in columns:
            n = _norm_name(c)
            if n in {"rt", "rtmean", "retentiontime", "rowretentiontime", "retime"} or ("retention" in n and "time" in n):
                return c
    if aliases is ID_ALIASES:
        for c in columns:
            n = _norm_name(c)
            if n in {"rowid", "clusterindex", "clusteridx", "featureid", "scan", "scannumber", "id"}:
                return c
    return None


def read_table_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Read CSV/TSV text, including PostMN MassQL's commented metadata preamble."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
            # PostMN MassQL Full Table exports start with metadata lines such as
            # '# PostMN MassQL export', '# task_id ...' and '# query ...'.
            # Those lines have a different number of tab-separated fields than
            # the actual data table, so delimiter inference must happen only
            # after they have been removed.
            if text.lstrip("\ufeff\r\n ").startswith("# PostMN MassQL export"):
                table_lines = [
                    line for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                if not table_lines:
                    raise ValueError("PostMN MassQL export contains metadata but no table header/data.")
                frame = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", low_memory=False)
            else:
                frame = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
                if len(frame.columns) <= 1:
                    # A few exports have an unambiguous extension but unusual delimiter parsing.
                    sep = "\t" if filename.lower().endswith((".tsv", ".txt")) else ","
                    frame = pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding, low_memory=False)
            frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
            return frame
        except Exception as exc:  # try other encodings
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not read {filename}. " + " | ".join(errors))


def standardize_feature_table(raw: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return standard columns feature_id, mz, rt plus original row fields."""
    if raw.empty:
        raise ValueError(f"{label} contains no rows.")
    cols = list(raw.columns)
    id_col = _find_column(cols, ID_ALIASES)
    mz_col = _find_column(cols, MZ_ALIASES)
    rt_col = _find_column(cols, RT_ALIASES)
    missing = []
    if id_col is None:
        missing.append("feature ID (e.g. row ID or cluster index)")
    if mz_col is None:
        missing.append("m/z (e.g. row m/z or parent mass)")
    if rt_col is None:
        missing.append("retention time (e.g. row retention time or RTMean)")
    if missing:
        raise ValueError(f"Could not identify {', '.join(missing)} in {label}. Available columns: {', '.join(cols[:30])}")

    frame = raw.copy()
    frame["__feature_id"] = frame[id_col].map(_clean_id)
    frame["__mz"] = pd.to_numeric(frame[mz_col], errors="coerce")
    frame["__rt"] = pd.to_numeric(frame[rt_col], errors="coerce")
    frame = frame[(frame["__feature_id"] != "") & frame["__mz"].notna() & (frame["__mz"] > 0)].copy()
    if frame.empty:
        raise ValueError(f"No rows with valid IDs and m/z values were found in {label}.")
    # Some inputs contain duplicated feature IDs. Keep one representative row per ID.
    frame = frame.drop_duplicates(subset=["__feature_id"], keep="first")
    standardized = pd.DataFrame({
        "feature_id": frame["__feature_id"].astype(str).values,
        "mz": frame["__mz"].astype(float).values,
        "rt": frame["__rt"].astype(float).values,
    })
    return standardized.reset_index(drop=True), {"id": id_col, "mz": mz_col, "rt": rt_col}


def neutral_mass_candidates(features: pd.DataFrame, adducts: dict[str, tuple[float, int]], polarity_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        for adduct_name, (mass_shift, charge) in adducts.items():
            neutral_mass = float(row.mz) * int(charge) - float(mass_shift)
            if neutral_mass > 0 and np.isfinite(neutral_mass):
                rows.append({
                    "feature_id": str(row.feature_id),
                    "mz": float(row.mz),
                    "rt": float(row.rt) if pd.notna(row.rt) else np.nan,
                    "neutral_mass": neutral_mass,
                    "adduct": adduct_name,
                    "polarity": polarity_label,
                })
    return pd.DataFrame(rows)


def match_features(
    positive: pd.DataFrame,
    negative: pd.DataFrame,
    positive_adducts: dict[str, tuple[float, int]],
    negative_adducts: dict[str, tuple[float, int]],
    mass_tolerance_ppm: float,
    rt_tolerance_min: float,
    target_polarity: str,
) -> pd.DataFrame:
    """Generate adduct-aware neutral-mass candidates, then keep best adduct pair per feature pair."""
    pos_c = neutral_mass_candidates(positive, positive_adducts, "positive")
    neg_c = neutral_mass_candidates(negative, negative_adducts, "negative")
    if pos_c.empty or neg_c.empty:
        return pd.DataFrame(columns=[
            "target_feature_id", "source_feature_id", "target_polarity", "source_polarity",
            "target_adduct", "source_adduct", "target_neutral_mass", "source_neutral_mass",
            "neutral_mass_error_da", "neutral_mass_error_ppm", "rt_difference_min", "candidate_adduct_pairs",
        ])

    if target_polarity == "positive":
        target_c, source_c = pos_c, neg_c
    else:
        target_c, source_c = neg_c, pos_c

    source_c = source_c.sort_values("neutral_mass").reset_index(drop=True)
    source_masses = source_c["neutral_mass"].to_numpy(dtype=float)
    raw_candidates: list[dict[str, Any]] = []

    # Search within a neutral-mass window in sorted source candidates.
    for t in target_c.itertuples(index=False):
        t_mass = float(t.neutral_mass)
        # ppm window is converted to Da around target neutral mass.
        tol_da = t_mass * float(mass_tolerance_ppm) / 1_000_000.0
        lo = int(np.searchsorted(source_masses, t_mass - tol_da, side="left"))
        hi = int(np.searchsorted(source_masses, t_mass + tol_da, side="right"))
        for s in source_c.iloc[lo:hi].itertuples(index=False):
            if pd.notna(t.rt) and pd.notna(s.rt):
                rt_error = abs(float(t.rt) - float(s.rt))
            else:
                rt_error = np.nan
            # RT compatibility is mandatory: both features must have RT values
            # and fall within the user-selected retention-time window.
            if not np.isfinite(rt_error) or rt_error > rt_tolerance_min:
                continue
            mass_error = float(t_mass) - float(s.neutral_mass)
            mean_mass = (float(t_mass) + float(s.neutral_mass)) / 2.0
            ppm_error = mass_error / mean_mass * 1_000_000.0 if mean_mass else np.nan
            raw_candidates.append({
                "target_feature_id": str(t.feature_id),
                "source_feature_id": str(s.feature_id),
                "target_polarity": target_polarity,
                "source_polarity": "negative" if target_polarity == "positive" else "positive",
                "target_adduct": str(t.adduct),
                "source_adduct": str(s.adduct),
                "target_neutral_mass": float(t.neutral_mass),
                "source_neutral_mass": float(s.neutral_mass),
                "neutral_mass_error_da": mass_error,
                "neutral_mass_error_ppm": ppm_error,
                "rt_difference_min": rt_error,
            })

    base_columns = [
        "target_feature_id", "source_feature_id", "target_polarity", "source_polarity",
        "target_adduct", "source_adduct", "target_neutral_mass", "source_neutral_mass",
        "neutral_mass_error_da", "neutral_mass_error_ppm", "rt_difference_min", "candidate_adduct_pairs",
    ]
    if not raw_candidates:
        return pd.DataFrame(columns=base_columns)

    all_pairs = pd.DataFrame(raw_candidates)
    pair_key = ["target_feature_id", "source_feature_id"]
    # Preserve the best-supported adduct hypothesis; report how many hypotheses agreed.
    all_pairs["__abs_ppm"] = all_pairs["neutral_mass_error_ppm"].abs()
    all_pairs["__abs_rt"] = all_pairs["rt_difference_min"].abs().fillna(np.inf)
    all_pairs = all_pairs.sort_values(["__abs_ppm", "__abs_rt"])
    hypothesis_counts = all_pairs.groupby(pair_key).size().rename("candidate_adduct_pairs")
    best = all_pairs.drop_duplicates(subset=pair_key, keep="first").drop(columns=["__abs_ppm", "__abs_rt"])
    best = best.merge(hypothesis_counts, on=pair_key, how="left")
    return best[base_columns].sort_values(["target_feature_id", "source_feature_id"]).reset_index(drop=True)


def graphml_bytes_to_graph(data: bytes) -> nx.Graph:
    try:
        return nx.read_graphml(io.BytesIO(data), force_multigraph=False)
    except TypeError:
        # Compatibility with older NetworkX versions without force_multigraph.
        return nx.read_graphml(io.BytesIO(data))


def graph_to_graphml_bytes(graph: nx.Graph) -> bytes:
    buffer = io.BytesIO()
    nx.write_graphml(graph, buffer, encoding="utf-8", prettyprint=True)
    return buffer.getvalue()


def node_feature_lookup(graph: nx.Graph) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    id_alias_norms = {_norm_name(a) for a in ID_ALIASES + ["name", "label", "cluster id", "row id", "feature id"]}
    for node_id, attrs in graph.nodes(data=True):
        candidates = [str(node_id)]
        for key, value in attrs.items():
            if _norm_name(key) in id_alias_norms:
                candidates.append(_clean_id(value))
        for candidate in candidates:
            cleaned = _clean_id(candidate)
            if cleaned:
                lookup.setdefault(cleaned, node_id)
    return lookup


def read_graphml_task_file(files: dict[str, bytes]) -> tuple[bytes, str]:
    candidates = [(p, b) for p, b in files.items() if p.lower().endswith(".graphml")]
    if not candidates:
        raise ValueError("No GraphML file was found in this GNPS2 task archive.")
    # Prefer the complete network including singleton nodes.
    candidates.sort(key=lambda pb: ("network_singletons.graphml" not in pb[0].lower(), "network.graphml" not in pb[0].lower(), len(pb[0])))
    return candidates[0][1], candidates[0][0]


def read_network_feature_table_from_task(files: dict[str, bytes]) -> tuple[pd.DataFrame, str, str]:
    """Prefer the GNPS2 node-level cluster summary; fall back to FBMN feature table."""
    paths = list(files.keys())
    cluster_summaries = [(p, b) for p, b in files.items() if p.lower().endswith("clustersummary_with_network.tsv")]
    if cluster_summaries:
        path, raw_bytes = sorted(cluster_summaries, key=lambda x: len(x[0]))[0]
        frame = read_table_bytes(raw_bytes, path)
        standardized, columns = standardize_feature_table(frame, path)
        return standardized, path, "node-level cluster summary"

    feature_tables = [(p, b) for p, b in files.items() if "featuretable" in p.lower() and p.lower().endswith((".csv", ".tsv"))]
    if feature_tables:
        # Prefer the main reformatted FBMN table over a presence-only derivative.
        feature_tables.sort(key=lambda x: ("presence" in x[0].lower(), "reformated" not in x[0].lower(), len(x[0])))
        path, raw_bytes = feature_tables[0]
        frame = read_table_bytes(raw_bytes, path)
        standardized, columns = standardize_feature_table(frame, path)
        return standardized, path, "feature table"
    raise ValueError("No usable feature table or clustersummary_with_network.tsv was found in the task archive.")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def fetch_gnps2_task(task_id: str) -> dict[str, bytes]:
    """Download selected output files from a public GNPS2 task without fetching raw-data archives."""
    task_id = task_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", task_id):
        raise ValueError("Task ID format looks unusual. Paste the task ID only, not the whole URL.")

    candidate_paths = [
        "nf_output/networking/network_singletons.graphml",
        "nf_output/networking/network.graphml",
        "nf_output/networking/clustersummary_with_network.tsv",
        "nf_output/clustering/featuretable_reformated.csv",
        "nf_output/clustering/featuretable_reformatted.csv",
        "nf_output/clustering/featuretable_reformatted_presence.csv",
    ]
    result: dict[str, bytes] = {}
    session = requests.Session()
    headers = {"User-Agent": "Cross-Polarity-Network-Enricher/0.1"}
    errors: list[str] = []
    for path in candidate_paths:
        url = f"https://gnps2.org/taskresult/{quote(task_id)}/file/{quote(path, safe='/')}"
        try:
            response = session.get(url, timeout=(5, 25), headers=headers)
            if response.status_code != 200:
                errors.append(f"{path}: HTTP {response.status_code}")
                continue
            content = response.content
            if not content:
                continue
            prefix = content[:300].lstrip().lower()
            # Missing files can return an HTML error page with HTTP 200.
            if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
                errors.append(f"{path}: server returned HTML rather than the requested file")
                continue
            if path.endswith(".graphml"):
                try:
                    nx.read_graphml(io.BytesIO(content))
                except Exception:
                    errors.append(f"{path}: response was not readable GraphML")
                    continue
            else:
                # Basic guard against error pages; table parsing is checked downstream.
                if b"\x00" in content[:1000]:
                    continue
            result[path] = content
        except requests.RequestException as exc:
            errors.append(f"{path}: {exc}")

    if not any(p.lower().endswith(".graphml") for p in result):
        raise ValueError(
            "Could not retrieve a readable GraphML file from this GNPS2 task. The task may be private, unfinished, "
            "or use different output paths. Try downloading its GraphML and quantification/cluster table manually. "
            + ("Details: " + "; ".join(errors[:4]) if errors else "")
        )
    if not any("clustersummary_with_network" in p.lower() or "featuretable" in p.lower() for p in result):
        raise ValueError(
            "A GraphML was found, but no recognized feature/cluster table could be retrieved. "
            "Please use Upload files and provide the positive and negative feature tables manually. "
            + ("Details: " + "; ".join(errors[:4]) if errors else "")
        )
    return result


def truthy_result(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    if not text or text in {"0", "0.0", "false", "no", "none", "nan", "fail", "failed", "not validated", "negative", "not detected"}:
        return False
    # Ignore a negated phrase before looking for positive tokens (e.g. "not validated").
    text_for_tokens = re.sub(r"\bnot\s+(validated|valid|passed|pass|hit)\b", "", text)
    # Exports may summarize several queries in one text cell, e.g. "query A: TRUE; query B: FALSE".
    positive_token = re.search(r"(?<![a-z])(true|yes|pass(?:ed)?|valid(?:ated)?|hit|positive)(?![a-z])", text_for_tokens)
    negative_only = re.search(r"(?<![a-z])(false|no|fail(?:ed)?|invalid|negative)(?![a-z])", text)
    if positive_token:
        return True
    if negative_only:
        return False
    if text in {"1", "1.0"}:
        return True
    return False


def infer_massql_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = list(frame.columns)
    id_candidates = [c for c in cols if _norm_name(c) in {_norm_name(x) for x in ["#Scan#", "scan", "scan number", "cluster index", "#ClusterIdx", "feature id", "row ID", "spectrum scan"]}]
    if not id_candidates:
        id_candidates = [c for c in cols if "scan" in _norm_name(c) or "clusterindex" in _norm_name(c) or "featureid" in _norm_name(c)]
    # Prioritize the actual PostMN MassQL validation field where present.
    status_candidates = [c for c in cols if _norm_name(c) == "queryvalidation"]
    if not status_candidates:
        status_candidates = [c for c in cols if any(token in _norm_name(c) for token in ["validated", "queryresult", "queryhit", "querypass", "massql"])]
    # Common exports can expose query names as separate boolean columns.
    if not status_candidates:
        status_candidates = [c for c in cols if "query" in _norm_name(c)]
    return (id_candidates[0] if id_candidates else None, status_candidates[0] if status_candidates else None)


def massql_validation_result(value: Any) -> bool:
    """Interpret PostMN MassQL query_validation values (query name = hit)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    failure_phrases = (
        "did not pass", "no selected query", "no query passed", "not validated",
        "not validated", "not detected", "no hit", "no match", "failed", "failure",
        "false", "negative", "not found", "did not match",
    )
    if any(phrase in lowered for phrase in failure_phrases):
        return False
    if lowered in {"0", "0.0", "no", "none", "nan", "fail", "invalid"}:
        return False
    # PostMN's Full Table stores a passing query's name (e.g. 'Seq_NL'), not
    # the literal value TRUE. Any non-empty, non-failure query_validation value
    # therefore represents one or more passing queries.
    return True


def massql_hit_map(frame: pd.DataFrame, id_col: str, status_col: str | None) -> tuple[dict[str, dict[str, Any]], int]:
    hits: dict[str, dict[str, Any]] = {}
    rows_with_id = 0
    for _, row in frame.iterrows():
        feature_id = _clean_id(row.get(id_col, ""))
        if not feature_id:
            # Some files store a USI/string that ends with scan:<number>.
            value = str(row.get(id_col, ""))
            match = re.search(r"(?:scan:|scan=)(\d+)", value, flags=re.IGNORECASE)
            if match:
                feature_id = match.group(1)
        if not feature_id:
            continue
        rows_with_id += 1
        if status_col:
            if _norm_name(status_col) == "queryvalidation":
                passed = massql_validation_result(row.get(status_col))
            else:
                passed = truthy_result(row.get(status_col))
        else:
            # The UI explicitly labels this option as treating every row with an ID as a hit.
            passed = True
        if not passed:
            continue
        detail = str(row.get(status_col, "hit")) if status_col else "hit"
        if feature_id not in hits:
            hits[feature_id] = {"count": 0, "details": set()}
        hits[feature_id]["count"] += 1
        if detail:
            hits[feature_id]["details"].add(detail)
    return hits, rows_with_id


def choose_node_value(mapping: dict[str, Any], feature_id: Any) -> Any | None:
    return mapping.get(_clean_id(feature_id))


def enrich_graph(
    graph: nx.Graph,
    matches: pd.DataFrame,
    target_features: pd.DataFrame,
    source_features: pd.DataFrame,
    target_polarity: str,
    target_massql_hits: dict[str, dict[str, Any]],
    source_massql_hits: dict[str, dict[str, Any]],
) -> tuple[nx.Graph, dict[str, Any]]:
    output = graph.copy()
    node_lookup = node_feature_lookup(output)
    target_ids_to_node: dict[str, Any] = {}
    for fid in target_features["feature_id"].astype(str):
        node_id = choose_node_value(node_lookup, fid)
        if node_id is not None:
            target_ids_to_node[fid] = node_id

    # Initialize explicit attributes on every node, making downstream styling easy.
    for node_id in output.nodes:
        attrs = output.nodes[node_id]
        attrs["CPNE_crosspolarity_match"] = "false"
        attrs["Matched cross polarities"] = "No"
        attrs["CPNE_massql_direct_hit"] = "false"
        attrs["CPNE_massql_transferred_hit"] = "false"
        attrs["CPNE_massql_hit"] = "false"
        attrs["CPNE_border_color"] = "#808080"
        attrs["CPNE_border_width"] = "1.0"
        attrs["CPNE_fill_color"] = "#D1D5DB"
        attrs["CPNE_match_count"] = 0
        attrs["CPNE_matched_source_feature_ids"] = ""
        attrs["CPNE_match_adduct_pairs"] = ""
        attrs["CPNE_best_mass_error_ppm"] = ""
        attrs["CPNE_best_rt_difference_min"] = ""
        attrs["CPNE_massql_details"] = ""

    per_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matches.to_dict(orient="records"):
        per_target[str(row["target_feature_id"])].append(row)

    unmatched_target_ids: list[str] = []
    for target_id, node_id in target_ids_to_node.items():
        node = output.nodes[node_id]
        records = per_target.get(target_id, [])
        if records:
            node["CPNE_crosspolarity_match"] = "true"
            node["Matched cross polarities"] = "Yes"
            node["CPNE_border_color"] = "#18854A"  # suggested green outline
            node["CPNE_border_width"] = "3.0"
            node["CPNE_match_count"] = len(records)
            node["CPNE_matched_source_feature_ids"] = ";".join(sorted({r["source_feature_id"] for r in records}))
            node["CPNE_match_adduct_pairs"] = ";".join(sorted({f'{r["target_adduct"]} / {r["source_adduct"]}' for r in records}))
            best = sorted(records, key=lambda r: (abs(float(r["neutral_mass_error_ppm"])), abs(float(r["rt_difference_min"])) if pd.notna(r["rt_difference_min"]) else float("inf")))[0]
            node["CPNE_best_mass_error_ppm"] = f'{float(best["neutral_mass_error_ppm"]):.5f}'
            node["CPNE_best_rt_difference_min"] = "" if pd.isna(best["rt_difference_min"]) else f'{float(best["rt_difference_min"]):.5f}'
        else:
            unmatched_target_ids.append(target_id)

        direct_hit = target_id in target_massql_hits
        if direct_hit:
            node["CPNE_massql_direct_hit"] = "true"
        transferred_details: set[str] = set()
        transferred = False
        for record in records:
            source_id = str(record["source_feature_id"])
            if source_id in source_massql_hits:
                transferred = True
                transferred_details.update(source_massql_hits[source_id]["details"])
        if transferred:
            node["CPNE_massql_transferred_hit"] = "true"
        if direct_hit or transferred:
            node["CPNE_massql_hit"] = "true"
            details = set()
            if direct_hit:
                details.update(target_massql_hits[target_id]["details"])
            details.update(transferred_details)
            node["CPNE_massql_details"] = ";".join(sorted(str(x) for x in details if str(x)))
            # If there is no cross-polarity match, use magenta to highlight a MassQL hit.
            if node["CPNE_crosspolarity_match"] != "true":
                node["CPNE_border_color"] = "#B12D87"
                node["CPNE_border_width"] = "3.0"
            # Preserve combined status in data: green outline + magenta fill key.
            node["CPNE_fill_color"] = "#E9B8D8"

    summary = {
        "target_nodes": output.number_of_nodes(),
        "target_features": len(target_features),
        "mapped_target_features": len(target_ids_to_node),
        "unmapped_target_features": len(target_features) - len(target_ids_to_node),
        "matched_target_features": int(matches["target_feature_id"].nunique()) if not matches.empty else 0,
        "matched_feature_pairs": len(matches),
        "target_direct_massql_hits": len(target_massql_hits),
        "source_massql_hits": len(source_massql_hits),
        "target_polarity": target_polarity,
    }
    return output, summary



def _numeric_node_mz(attrs: dict[str, Any]) -> float | None:
    """Find a node's precursor m/z in common GraphML node attributes."""
    preferred = {
        _norm_name(x) for x in [
            "precursor m/z", "precursor mass", "parent mass", "row m/z",
            "mz", "m/z", "specmz", "observed mz", "precursormz",
        ]
    }
    # Prefer explicit precursor/mz names over generic 'mass' labels.
    ordered = sorted(attrs.items(), key=lambda item: (_norm_name(item[0]) not in preferred, str(item[0])))
    for key, value in ordered:
        normalized = _norm_name(key)
        if normalized not in preferred:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def tag_edges_by_delta_mass(
    graph: nx.Graph,
    target_features: pd.DataFrame,
    enabled: bool,
    target_delta_mass_da: float | None = None,
    tolerance_da: float | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Tag existing network edges whose connected-node precursor m/z gap matches a target.

    The comparison uses the absolute precursor m/z difference between the two nodes
    connected by each edge. Feature-table m/z values are preferred; GraphML node
    attributes are used as a fallback.
    """
    output = graph.copy()
    summary: dict[str, Any] = {
        "edge_delta_mass_tagging_enabled": bool(enabled),
        "edge_count": int(output.number_of_edges()),
        "edges_checked_for_delta_mass": 0,
        "edges_matching_delta_mass": 0,
        "edges_missing_precursor_mz": 0,
        "edge_target_delta_mass_da": float(target_delta_mass_da) if enabled and target_delta_mass_da is not None else None,
        "edge_delta_mass_tolerance_da": float(tolerance_da) if enabled and tolerance_da is not None else None,
    }
    if not enabled:
        return output, summary
    if target_delta_mass_da is None or tolerance_da is None:
        raise ValueError("Provide both a target delta mass and an allowed error to tag edges.")
    if target_delta_mass_da < 0 or tolerance_da < 0:
        raise ValueError("Target delta mass and allowed error must be non-negative.")

    # Start with precursor masses already stored on GraphML nodes.
    node_mz: dict[Any, float] = {}
    for node_id, attrs in output.nodes(data=True):
        value = _numeric_node_mz(attrs)
        if value is not None:
            node_mz[node_id] = value

    # Prefer the target feature table when a feature ID can be mapped to a network node.
    lookup = node_feature_lookup(output)
    for row in target_features.itertuples(index=False):
        node_id = lookup.get(_clean_id(row.feature_id))
        if node_id is None:
            continue
        try:
            mz = float(row.mz)
        except (TypeError, ValueError):
            continue
        if np.isfinite(mz) and mz > 0:
            node_mz[node_id] = mz

    matched_edges = 0
    checked_edges = 0
    missing_edges = 0
    target_text = f"{float(target_delta_mass_da):.8f}"
    tolerance_text = f"{float(tolerance_da):.8f}"

    for node_a, node_b, attrs in output.edges(data=True):
        mz_a = node_mz.get(node_a)
        mz_b = node_mz.get(node_b)
        attrs["CPNE_target_precursor_delta_mass_da"] = target_text
        attrs["CPNE_precursor_delta_mass_tolerance_da"] = tolerance_text
        attrs["Selected precursor Δm (Da)"] = target_text
        attrs["Allowed precursor Δm error (Da)"] = tolerance_text

        if mz_a is None or mz_b is None:
            attrs["CPNE_observed_precursor_delta_mass_da"] = ""
            attrs["CPNE_precursor_delta_mass_error_da"] = ""
            attrs["CPNE_delta_mass_match"] = "unknown"
            attrs["Observed precursor Δm (Da)"] = ""
            attrs["Precursor Δm error (Da)"] = ""
            attrs["Matches selected precursor Δm"] = "Unknown"
            missing_edges += 1
            continue

        observed = abs(float(mz_a) - float(mz_b))
        error = observed - float(target_delta_mass_da)
        is_match = abs(error) <= float(tolerance_da) + 1e-12
        checked_edges += 1
        matched_edges += int(is_match)
        attrs["CPNE_observed_precursor_delta_mass_da"] = f"{observed:.8f}"
        attrs["CPNE_precursor_delta_mass_error_da"] = f"{error:+.8f}"
        attrs["CPNE_delta_mass_match"] = "true" if is_match else "false"
        attrs["Observed precursor Δm (Da)"] = f"{observed:.8f}"
        attrs["Precursor Δm error (Da)"] = f"{error:+.8f}"
        attrs["Matches selected precursor Δm"] = "Yes" if is_match else "No"

    summary["edges_checked_for_delta_mass"] = checked_edges
    summary["edges_matching_delta_mass"] = matched_edges
    summary["edges_missing_precursor_mz"] = missing_edges
    return output, summary


def graph_edge_table(graph: nx.Graph) -> pd.DataFrame:
    """Flatten graph edges and their attributes to a downloadable table."""
    rows: list[dict[str, Any]] = []
    for source, target, attrs in graph.edges(data=True):
        row: dict[str, Any] = {"source_node": str(source), "target_node": str(target)}
        for key, value in attrs.items():
            row[str(key)] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["source_node", "target_node"])
    return pd.DataFrame(rows)

def cytoscape_style_xml() -> bytes:
    """Return a Cytoscape Style XML that maps the CPNE node attributes to visible styling."""
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<vizmap documentVersion="3.0" id="VizMap-CPNE">
  <visualStyle name="CPNE Cross-Polarity and MassQL">
    <network>
      <visualProperty name="NETWORK_BACKGROUND_PAINT" default="#FFFFFF"/>
    </network>
    <node>
      <visualProperty name="NODE_LABEL" default="">
        <passthroughMapping attributeType="string" attributeName="label"/>
      </visualProperty>
      <visualProperty name="NODE_SHAPE" default="ELLIPSE"/>
      <visualProperty name="NODE_SIZE" default="32.0"/>
      <visualProperty name="NODE_FILL_COLOR" default="#D1D5DB">
        <passthroughMapping attributeType="string" attributeName="CPNE_fill_color"/>
      </visualProperty>
      <visualProperty name="NODE_BORDER_PAINT" default="#808080">
        <passthroughMapping attributeType="string" attributeName="CPNE_border_color"/>
      </visualProperty>
      <visualProperty name="NODE_BORDER_WIDTH" default="1.0">
        <discreteMapping attributeType="string" attributeName="CPNE_border_width">
          <discreteMappingEntry attributeValue="1.0" value="1.0"/>
          <discreteMappingEntry attributeValue="3.0" value="3.0"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty name="NODE_BORDER_STROKE" default="SOLID"/>
      <visualProperty name="NODE_LABEL_FONT_SIZE" default="12"/>
      <visualProperty name="NODE_LABEL_COLOR" default="#333333"/>
      <visualProperty name="NODE_VISIBLE" default="true"/>
    </node>
    <edge>
      <visualProperty name="EDGE_UNSELECTED_PAINT" default="#C8C8C8"/>
      <visualProperty name="EDGE_WIDTH" default="1.0"/>
      <visualProperty name="EDGE_VISIBLE" default="true"/>
    </edge>
  </visualStyle>
</vizmap>
"""
    return xml.encode("utf-8")


def _prepare_gnps2_pair(pos_files: dict[str, bytes], neg_files: dict[str, bytes]) -> dict[str, Any]:
    """Parse the selected outputs for both example GNPS2 tasks into one reusable state."""
    pos_features, pos_path, pos_kind = read_network_feature_table_from_task(pos_files)
    neg_features, neg_path, neg_kind = read_network_feature_table_from_task(neg_files)
    pos_graph_bytes, pos_graph_path = read_graphml_task_file(pos_files)
    neg_graph_bytes, neg_graph_path = read_graphml_task_file(neg_files)
    # Parse both GraphML files up front so a malformed file is caught before caching.
    pos_graph = graphml_bytes_to_graph(pos_graph_bytes)
    neg_graph = graphml_bytes_to_graph(neg_graph_bytes)
    return {
        "positive_features": pos_features,
        "negative_features": neg_features,
        "positive_graph_bytes": pos_graph_bytes,
        "negative_graph_bytes": neg_graph_bytes,
        "positive_graph": pos_graph,
        "negative_graph": neg_graph,
        "source_names": {
            "positive": f"{DEMO_POSITIVE_TASK_ID}: {pos_path} ({pos_kind})",
            "negative": f"{DEMO_NEGATIVE_TASK_ID}: {neg_path} ({neg_kind})",
            "graph_positive": pos_graph_path,
            "graph_negative": neg_graph_path,
        },
    }


def _save_demo_task_cache(pos_files: dict[str, bytes], neg_files: dict[str, bytes]) -> None:
    """Persist the real task outputs locally after a successful fetch for later offline demos."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    metadata = {
        "positive_task_id": DEMO_POSITIVE_TASK_ID,
        "negative_task_id": DEMO_NEGATIVE_TASK_ID,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Selected GraphML and node/feature tables fetched for CPNE's example tasks.",
    }
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        for path, content in pos_files.items():
            archive.writestr(f"positive/{path}", content)
        for path, content in neg_files.items():
            archive.writestr(f"negative/{path}", content)
    DEMO_TASK_CACHE_PATH.write_bytes(buffer.getvalue())


def _read_demo_task_cache() -> tuple[dict[str, bytes], dict[str, bytes]] | None:
    """Load a previous real-task snapshot, if one exists and matches the bundled task IDs."""
    if not DEMO_TASK_CACHE_PATH.exists():
        return None
    try:
        with zipfile.ZipFile(DEMO_TASK_CACHE_PATH, "r") as archive:
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            if (
                metadata.get("positive_task_id") != DEMO_POSITIVE_TASK_ID
                or metadata.get("negative_task_id") != DEMO_NEGATIVE_TASK_ID
            ):
                return None
            pos_files: dict[str, bytes] = {}
            neg_files: dict[str, bytes] = {}
            for name in archive.namelist():
                if name.startswith("positive/") and not name.endswith("/"):
                    pos_files[name[len("positive/"):]] = archive.read(name)
                elif name.startswith("negative/") and not name.endswith("/"):
                    neg_files[name[len("negative/"):]] = archive.read(name)
        if not pos_files or not neg_files:
            return None
        return pos_files, neg_files
    except Exception:
        return None


def _load_bundled_offline_demo() -> dict[str, Any]:
    """Load the small, clearly labelled synthetic fallback shipped with CPNE."""
    pos_path = EXAMPLES_DIR / "positive_quant.csv"
    neg_path = EXAMPLES_DIR / "negative_quant.csv"
    pos_graph_path = EXAMPLES_DIR / "positive_network.graphml"
    neg_graph_path = EXAMPLES_DIR / "negative_network.graphml"
    pos_raw = read_table_bytes(pos_path.read_bytes(), pos_path.name)
    neg_raw = read_table_bytes(neg_path.read_bytes(), neg_path.name)
    pos_features, _ = standardize_feature_table(pos_raw, pos_path.name)
    neg_features, _ = standardize_feature_table(neg_raw, neg_path.name)
    pos_graph_bytes = pos_graph_path.read_bytes()
    neg_graph_bytes = neg_graph_path.read_bytes()
    return {
        "positive_features": pos_features,
        "negative_features": neg_features,
        "positive_graph_bytes": pos_graph_bytes,
        "negative_graph_bytes": neg_graph_bytes,
        "positive_graph": graphml_bytes_to_graph(pos_graph_bytes),
        "negative_graph": graphml_bytes_to_graph(neg_graph_bytes),
        "source_names": {
            "positive": "Bundled offline fallback: synthetic positive feature table",
            "negative": "Bundled offline fallback: synthetic negative feature table",
            "graph_positive": "examples/positive_network.graphml (synthetic)",
            "graph_negative": "examples/negative_network.graphml (synthetic)",
        },
    }


def _load_example_task_data() -> tuple[dict[str, Any], str, str | None]:
    """Prefer live GNPS2 tasks, then a previously saved real snapshot, then a synthetic fallback."""
    live_error: str | None = None
    try:
        pos_files = fetch_gnps2_task(DEMO_POSITIVE_TASK_ID)
        neg_files = fetch_gnps2_task(DEMO_NEGATIVE_TASK_ID)
        state = _prepare_gnps2_pair(pos_files, neg_files)
        try:
            _save_demo_task_cache(pos_files, neg_files)
            state["cache_saved"] = True
        except Exception as exc:
            # A read-only installation can still use the fetched data for this session.
            state["cache_saved"] = False
            state["cache_save_error"] = str(exc)
        return state, "live", None
    except Exception as exc:
        live_error = str(exc)

    cached = _read_demo_task_cache()
    if cached is not None:
        try:
            state = _prepare_gnps2_pair(*cached)
            state["cache_saved"] = True
            state["cache_path"] = str(DEMO_TASK_CACHE_PATH.name)
            return state, "cached", live_error
        except Exception as exc:
            live_error = f"Live fetch failed ({live_error}); cached snapshot could not be parsed ({exc})."

    state = _load_bundled_offline_demo()
    return state, "offline_fallback", live_error


def _activate_demo() -> None:
    """Preload the user's example GNPS2 tasks and matching parameters from the demo button."""
    st.session_state["input_mode"] = "Enter GNPS2 task IDs"
    st.session_state["network_direction"] = "Positive network ← Negative data"
    st.session_state["positive_task_id"] = DEMO_POSITIVE_TASK_ID
    st.session_state["negative_task_id"] = DEMO_NEGATIVE_TASK_ID
    st.session_state["positive_adduct_selection"] = ["[M+H]+", "[M+Na]+", "[M+NH4]+"]
    st.session_state["negative_adduct_selection"] = ["[M-H]-", "[M-H2O-H]-", "[M+FA-H]-", "[M+Na-2H]-"]
    st.session_state["pos_custom_enabled"] = False
    st.session_state["neg_custom_enabled"] = False
    st.session_state["mass_tolerance_ppm_input"] = "10.0"
    st.session_state["rt_tolerance_min_input"] = "0.20"
    st.session_state["tag_edge_delta_mass"] = True
    st.session_state["edge_delta_mass_text"] = "42.047"
    st.session_state["edge_delta_mass_tolerance_text"] = "0.001"
    st.session_state["cpne_demo_mode"] = True
    st.session_state.pop("cpne_demo_data", None)
    st.session_state.pop("cpne_demo_origin", None)
    st.session_state.pop("cpne_demo_error", None)
    st.session_state.pop("cpne_result", None)


def _load_bundled_massql(polarity: str, offline_fallback: bool) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, str]:
    """Read a bundled MassQL TSV; use synthetic hits only with the synthetic offline example."""
    if offline_fallback:
        filename = f"offline_demo_{polarity}_massql_full_table.tsv"
    else:
        filename = f"provided_postmn_{polarity}.tsv"
    path = EXAMPLES_DIR / filename
    frame = read_table_bytes(path.read_bytes(), path.name)
    id_col, status_col = infer_massql_columns(frame)
    if id_col is None:
        raise ValueError(f"Could not identify a scan/node ID column in bundled file {filename}.")
    hits, _ = massql_hit_map(frame, id_col, status_col)
    return hits, frame, filename


def _show_bundled_massql_preview(polarity: str, frame: pd.DataFrame, filename: str) -> None:
    auto_id, auto_status = infer_massql_columns(frame)
    n_hits = 0
    if auto_id:
        n_hits = len(massql_hit_map(frame, auto_id, auto_status)[0])
    st.caption(f"Bundled {polarity}-mode MassQL export: {filename} • {len(frame):,} rows • {n_hits:,} unique hit IDs detected")
    with st.expander(f"Preview bundled {polarity}-mode MassQL results"):
        st.dataframe(frame.head(10), use_container_width=True, hide_index=True)


def _read_graphml_upload(uploaded_file: Any) -> tuple[nx.Graph, bytes]:
    data = uploaded_file.getvalue()
    return graphml_bytes_to_graph(data), data


def _adduct_selection_ui() -> tuple[dict[str, tuple[float, int]], dict[str, tuple[float, int]]]:
    st.subheader("3. Adduct settings")
    st.caption("Mass shifts are relative to the neutral molecule M. Charge magnitude is used to convert observed m/z to neutral mass.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Positive mode")
        if "positive_adduct_selection" not in st.session_state:
            st.session_state["positive_adduct_selection"] = ["[M+H]+"]
        pos_selected = st.multiselect("Select positive-mode adducts", list(POSITIVE_ADDUCTS), key="positive_adduct_selection")
        pos_custom_enabled = st.checkbox("Add a custom positive adduct", key="pos_custom_enabled")
        pos_custom: dict[str, tuple[float, int]] = {}
        if pos_custom_enabled:
            pos_name = st.text_input("Custom positive adduct name", value="[M+Li]+", key="pos_custom_name").strip()
            pos_shift = st.number_input("Total ion mass shift (Da)", value=7.015455, format="%.6f", key="pos_custom_shift")
            pos_charge = st.number_input("Charge magnitude |z|", min_value=1, max_value=10, value=1, step=1, key="pos_custom_charge")
            if pos_name:
                pos_custom[pos_name] = (float(pos_shift), int(pos_charge))
    with col2:
        st.markdown("#### Negative mode")
        if "negative_adduct_selection" not in st.session_state:
            st.session_state["negative_adduct_selection"] = ["[M-H]-"]
        neg_selected = st.multiselect("Select negative-mode adducts", list(NEGATIVE_ADDUCTS), key="negative_adduct_selection")
        neg_custom_enabled = st.checkbox("Add a custom negative adduct", key="neg_custom_enabled")
        neg_custom: dict[str, tuple[float, int]] = {}
        if neg_custom_enabled:
            neg_name = st.text_input("Custom negative adduct name", value="[M+TFA-H]-", key="neg_custom_name").strip()
            neg_shift = st.number_input("Total ion mass shift (Da)", value=112.985586, format="%.6f", key="neg_custom_shift")
            neg_charge = st.number_input("Charge magnitude |z|", min_value=1, max_value=10, value=1, step=1, key="neg_custom_charge")
            if neg_name:
                neg_custom[neg_name] = (float(neg_shift), int(neg_charge))
    pos = {name: POSITIVE_ADDUCTS[name] for name in pos_selected}
    neg = {name: NEGATIVE_ADDUCTS[name] for name in neg_selected}
    pos.update(pos_custom)
    neg.update(neg_custom)
    return pos, neg


def _massql_upload_ui(label: str, polarity: str) -> tuple[dict[str, dict[str, Any]], pd.DataFrame | None, str | None]:
    upload = st.file_uploader(label, type=["tsv", "csv", "txt"], key=f"massql_{polarity}")
    if upload is None:
        return {}, None, None
    try:
        frame = read_table_bytes(upload.getvalue(), upload.name)
    except Exception as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return {}, None, None
    st.caption(f"{polarity.title()} MassQL file: {len(frame):,} rows × {len(frame.columns):,} columns")
    auto_id, auto_status = infer_massql_columns(frame)
    columns = list(frame.columns)
    id_index = columns.index(auto_id) if auto_id in columns else 0
    id_col = st.selectbox(f"{polarity.title()} MassQL scan/node ID column", columns, index=id_index, key=f"massql_id_{polarity}")
    status_options = ["No status filter (treat every row as a hit)"] + columns
    status_default = columns.index(auto_status) + 1 if auto_status in columns else 0
    status_choice = st.selectbox(f"{polarity.title()} MassQL hit/validation column", status_options, index=status_default, key=f"massql_status_{polarity}")
    status_col = None if status_choice.startswith("No status filter") else status_choice
    hits, rows_with_id = massql_hit_map(frame, id_col, status_col)
    if status_col is None:
        st.warning("No status filter selected: every row with an ID will be treated as a MassQL hit.")
    elif not hits:
        st.warning("No positive MassQL hits were detected from that column. Check that the selected column encodes TRUE/PASS/HIT (or choose no status filter).")
    else:
        st.success(f"Detected {len(hits):,} unique hit IDs across {rows_with_id:,} rows with IDs.")
    with st.expander(f"Preview {polarity} MassQL results"):
        st.dataframe(frame.head(10), use_container_width=True)
    return hits, frame, upload.name


def _show_how_to_use() -> None:
    """Open the in-app quick-start guide as a separate view."""
    st.session_state["show_how_to_use"] = True


def _show_main() -> None:
    """Return from the guide to the main CPNE workflow."""
    st.session_state["show_how_to_use"] = False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=str(LOGO_PATH), layout="wide")
    logo_col, title_col = st.columns([0.13, 0.87])
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=100)
    with title_col:
        st.title("Cross-Polarity Network Enricher")
        st.markdown(
            "Match LC–MS features across positive and negative ionization modes, then enrich a molecular network with cross-polarity and optional PostMN MassQL information. "
            "For creating PostMN MassQL .tsv files, go here: [https://apps.gnps2.org/postmn_massql](https://apps.gnps2.org/postmn_massql)."
        )
    st.caption(
        "CPNE v0.7 • Local runs process files on your computer. In a hosted deployment, uploaded files are processed on the app server; "
        "do not upload confidential or restricted data to a public demo. Runtime-generated GNPS2 cache files may not persist after cloud restarts."
    )

    if "show_how_to_use" not in st.session_state:
        st.session_state["show_how_to_use"] = False

    # Present the instructions as a separate view with an obvious route back.
    if st.session_state["show_how_to_use"]:
        st.divider()
        st.button("← Main", type="primary", on_click=_show_main, use_container_width=True, key="return_to_main")
        st.title("How to use CPNE")
        with st.container(border=True):
            if HOWTO_PATH.exists():
                st.markdown(HOWTO_PATH.read_text(encoding="utf-8"))
            else:
                st.warning("HOW_TO_USE.md is missing from this installation. See the bundled README for basic instructions.")
        st.stop()

    demo_col, guide_col, demo_text_col = st.columns([1.05, 0.9, 2.05])
    with demo_col:
        st.button("Load example files", type="secondary", on_click=_activate_demo, use_container_width=True)
    with guide_col:
        st.button("How to use CPNE", type="secondary", on_click=_show_how_to_use, use_container_width=True)
    with demo_text_col:
        st.caption("Preloads the demo tasks, selected adducts, matching parameters, edge Δm tag, and bundled PostMN MassQL results.")
    st.divider()

    st.subheader("1. Data input")
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Upload files"
    input_mode = st.radio("How would you like to provide your data?", ["Upload files", "Enter GNPS2 task IDs"], horizontal=True, key="input_mode")

    st.subheader("2. Network enrichment")
    if "network_direction" not in st.session_state:
        st.session_state["network_direction"] = "Positive network ← Negative data"
    network_direction = st.radio(
        "Which network should be enriched?",
        ["Positive network ← Negative data", "Negative network ← Positive data"],
        horizontal=True,
        key="network_direction",
    )
    target_polarity = "positive" if network_direction.startswith("Positive") else "negative"
    source_polarity = "negative" if target_polarity == "positive" else "positive"
    st.info(f"The **{target_polarity}-mode network** will be retained and enriched with information from the **{source_polarity}-mode data**.")

    pos_adducts, neg_adducts = _adduct_selection_ui()
    st.divider()
    st.subheader("4. Matching parameters")
    if "mass_tolerance_ppm_input" not in st.session_state:
        st.session_state["mass_tolerance_ppm_input"] = "10.0"
    if "rt_tolerance_min_input" not in st.session_state:
        st.session_state["rt_tolerance_min_input"] = "0.20"
    p1, p2 = st.columns(2)
    with p1:
        mass_tolerance_text = st.text_input("Neutral-mass tolerance (ppm)", help="Use a period as the decimal separator, e.g. 10.0.", key="mass_tolerance_ppm_input")
    with p2:
        rt_tolerance_text = st.text_input("Retention-time window (min)", help="Use a period as the decimal separator, e.g. 0.20.", key="rt_tolerance_min_input")
    st.caption("Retention-time compatibility is always required: both features must have an RT value and fall within this window. This assumes comparable chromatographic conditions for both polarity runs.")

    def parse_dot_decimal(value: str, label: str, minimum: float, maximum: float) -> float | None:
        raw = value.strip()
        try:
            parsed = float(raw)
            if not np.isfinite(parsed) or parsed < minimum or parsed > maximum:
                raise ValueError
            return parsed
        except (TypeError, ValueError):
            st.error(f"{label} must be a number between {minimum:g} and {maximum:g}. Use a period for decimals (for example, 0.20), not a comma.")
            return None

    mass_tolerance_ppm = parse_dot_decimal(mass_tolerance_text, "Neutral-mass tolerance", 0.1, 1000.0)
    rt_tolerance_min = parse_dot_decimal(rt_tolerance_text, "Retention-time window", 0.0, 100.0)
    st.caption("The RT filter assumes comparable chromatographic conditions for the two polarity runs. Neutral mass is calculated from each selected adduct and charge magnitude.")

    st.divider()
    st.subheader("5. Optional edge tagging")
    tag_edge_delta_mass = st.checkbox(
        "Tag edges matching a specific precursor mass difference?",
        help="Checks the absolute difference between precursor m/z values of the two nodes connected by each network edge. This is not a fragment neutral-loss search.",
        key="tag_edge_delta_mass",
    )
    edge_delta_mass_da: float | None = None
    edge_delta_mass_tolerance_da: float | None = None
    if tag_edge_delta_mass:
        if "edge_delta_mass_text" not in st.session_state:
            st.session_state["edge_delta_mass_text"] = "25.396"
        if "edge_delta_mass_tolerance_text" not in st.session_state:
            st.session_state["edge_delta_mass_tolerance_text"] = "0.001"
        e1, e2 = st.columns(2)
        with e1:
            edge_delta_mass_text = st.text_input(
                "Target precursor Δm (Da)", key="edge_delta_mass_text",
                help="Enter the desired absolute difference between connected-node precursor m/z values, e.g. 25.396.",
            )
        with e2:
            edge_delta_mass_tolerance_text = st.text_input(
                "Allowed error ± (Da)", key="edge_delta_mass_tolerance_text",
                help="An allowed error of 0.001 Da tags observed differences from target − 0.001 to target + 0.001 Da.",
            )
        edge_delta_mass_da = parse_dot_decimal(edge_delta_mass_text, "Target precursor Δm", 0.0, 100000.0)
        edge_delta_mass_tolerance_da = parse_dot_decimal(edge_delta_mass_tolerance_text, "Allowed precursor Δm error", 0.0, 1000.0)
        st.caption("Matching edges will receive GraphML edge-table attributes including the observed Δm, signed error, and a Yes/No tag. Edges without recoverable precursor m/z values are marked Unknown.")

    pos_features: pd.DataFrame | None = None
    neg_features: pd.DataFrame | None = None
    graph: nx.Graph | None = None
    graph_bytes: bytes | None = None
    source_names: dict[str, str] = {}
    input_error: str | None = None

    st.divider()
    st.subheader("6. Input data")
    active_demo = False
    demo_origin: str | None = None
    demo_state: dict[str, Any] | None = None

    if input_mode == "Upload files":
        c1, c2 = st.columns(2)
        with c1:
            pos_upload = st.file_uploader("Positive-mode feature/quantification table (.csv or .tsv)", type=["csv", "tsv", "txt"], key="positive_quant")
            neg_upload = st.file_uploader("Negative-mode feature/quantification table (.csv or .tsv)", type=["csv", "tsv", "txt"], key="negative_quant")
        graph_upload = st.file_uploader(f"Target network GraphML ({target_polarity}-mode network)", type=["graphml", "xml"], key="target_graphml")
        if pos_upload and neg_upload and graph_upload:
            try:
                pos_raw = read_table_bytes(pos_upload.getvalue(), pos_upload.name)
                neg_raw = read_table_bytes(neg_upload.getvalue(), neg_upload.name)
                pos_features, pos_cols = standardize_feature_table(pos_raw, pos_upload.name)
                neg_features, neg_cols = standardize_feature_table(neg_raw, neg_upload.name)
                graph, graph_bytes = _read_graphml_upload(graph_upload)
                source_names = {"positive": pos_upload.name, "negative": neg_upload.name, "graph": graph_upload.name}
                st.success(f"Loaded {len(pos_features):,} positive features, {len(neg_features):,} negative features, and a network with {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges.")
                with st.expander("Detected input columns"):
                    st.write({"Positive": pos_cols, "Negative": neg_cols})
            except Exception as exc:
                input_error = str(exc)
                st.error(input_error)
    else:
        task_c1, task_c2 = st.columns(2)
        with task_c1:
            positive_task_id = st.text_input("Positive-mode GNPS2 task ID", placeholder="Paste task ID only", key="positive_task_id")
        with task_c2:
            negative_task_id = st.text_input("Negative-mode GNPS2 task ID", placeholder="Paste task ID only", key="negative_task_id")

        active_demo = bool(
            st.session_state.get("cpne_demo_mode", False)
            and positive_task_id.strip() == DEMO_POSITIVE_TASK_ID
            and negative_task_id.strip() == DEMO_NEGATIVE_TASK_ID
        )

        if active_demo:
            st.caption("Example profile selected. CPNE tries these public GNPS2 jobs first, checks for a saved snapshot, and falls back to bundled offline examples if needed. Runtime-written files may not persist on hosted platforms.")
            if st.button("Retry fetching the example from GNPS2", type="secondary", key="retry_demo_gnps2"):
                st.session_state.pop("cpne_demo_data", None)
                st.session_state.pop("cpne_demo_origin", None)
                st.session_state.pop("cpne_demo_error", None)
                st.rerun()

            if "cpne_demo_data" not in st.session_state:
                with st.spinner("Trying to retrieve the configured GNPS2 tasks; checking local demo cache if needed…"):
                    loaded, origin, load_error = _load_example_task_data()
                st.session_state["cpne_demo_data"] = loaded
                st.session_state["cpne_demo_origin"] = origin
                st.session_state["cpne_demo_error"] = load_error

            demo_state = st.session_state["cpne_demo_data"]
            demo_origin = st.session_state.get("cpne_demo_origin", "offline_fallback")
            pos_features = demo_state["positive_features"]
            neg_features = demo_state["negative_features"]
            graph_bytes = demo_state[f"{target_polarity}_graph_bytes"]
            graph = demo_state[f"{target_polarity}_graph"]
            source_names = dict(demo_state["source_names"])
            source_names["graph"] = source_names[f"graph_{target_polarity}"]

            if demo_origin == "live":
                st.success(f"Loaded the real GNPS2 task outputs: {len(pos_features):,} positive features, {len(neg_features):,} negative features, and {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges in the {target_polarity} target network.")
                if demo_state.get("cache_saved"):
                    st.caption(f"A snapshot was saved to `examples/{DEMO_TASK_CACHE_PATH.name}` in this app runtime. On hosted platforms, runtime file storage may not persist after a restart.")
                elif demo_state.get("cache_save_error"):
                    st.warning(f"Live data loaded, but CPNE could not save the local snapshot in this folder: {demo_state['cache_save_error']}")
            elif demo_origin == "cached":
                st.info(f"GNPS2 could not be reached, so CPNE loaded a previously saved real-task snapshot. Positive features: {len(pos_features):,}; negative features: {len(neg_features):,}; target graph: {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges. On hosted platforms, this runtime cache may disappear after a restart.")
            else:
                st.warning("GNPS2 could not be reached and no real-task snapshot is stored in this installation yet. CPNE is using the bundled **synthetic offline fallback** for the network and feature tables, so you can explore the workflow. Your real PostMN MassQL exports are bundled too, but they are not mapped onto these synthetic nodes; synthetic MassQL examples are used for this offline run.")
                last_error = st.session_state.get("cpne_demo_error")
                if last_error:
                    with st.expander("Why the live task fetch was unavailable"):
                        st.code(str(last_error))
            st.success(f"Ready: {len(pos_features):,} positive feature rows; {len(neg_features):,} negative feature rows; {graph.number_of_nodes():,} target nodes; {graph.number_of_edges():,} target edges.")
        else:
            st.caption("For ordinary task input, CPNE downloads selected output files from public GNPS2 results. Private tasks or unavailable output paths may need the file-upload route instead.")
            if positive_task_id.strip() and negative_task_id.strip():
                if st.button("Fetch GNPS2 task results", type="secondary"):
                    try:
                        with st.spinner("Downloading positive-mode task outputs…"):
                            pos_files = fetch_gnps2_task(positive_task_id.strip())
                        with st.spinner("Downloading negative-mode task outputs…"):
                            neg_files = fetch_gnps2_task(negative_task_id.strip())
                        pos_features, pos_path, pos_kind = read_network_feature_table_from_task(pos_files)
                        neg_features, neg_path, neg_kind = read_network_feature_table_from_task(neg_files)
                        target_files = pos_files if target_polarity == "positive" else neg_files
                        graph_bytes, graph_path = read_graphml_task_file(target_files)
                        graph = graphml_bytes_to_graph(graph_bytes)
                        st.session_state["gnps_fetched_state"] = {
                            "positive_features": pos_features,
                            "negative_features": neg_features,
                            "graph_bytes": graph_bytes,
                            "graph": graph,
                            "source_names": {"positive": f"{positive_task_id}: {pos_path} ({pos_kind})", "negative": f"{negative_task_id}: {neg_path} ({neg_kind})", "graph": graph_path},
                            "positive_task_id": positive_task_id.strip(),
                            "negative_task_id": negative_task_id.strip(),
                            "target_polarity": target_polarity,
                        }
                        st.success(f"Fetched both task outputs. Positive: {len(pos_features):,} feature rows; negative: {len(neg_features):,} feature rows; target network: {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges.")
                    except Exception as exc:
                        st.session_state.pop("gnps_fetched_state", None)
                        st.error(f"Could not fetch/parse the GNPS2 task(s): {exc}")
            task_state = st.session_state.get("gnps_fetched_state")
            if task_state:
                state_matches_inputs = (
                    task_state.get("positive_task_id") == positive_task_id.strip()
                    and task_state.get("negative_task_id") == negative_task_id.strip()
                    and task_state.get("target_polarity") == target_polarity
                )
                if state_matches_inputs:
                    pos_features = task_state["positive_features"]
                    neg_features = task_state["negative_features"]
                    graph_bytes = task_state["graph_bytes"]
                    graph = task_state["graph"]
                    source_names = task_state["source_names"]
                    st.success("Using the fetched task files currently shown above.")
                else:
                    st.warning("The task IDs or target polarity changed since the last fetch. Click ‘Fetch GNPS2 task results’ again before matching.")
    st.divider()
    st.subheader("7. Optional PostMN MassQL annotations")
    st.write("MassQL hits can annotate direct hits on the target network and transfer source-polarity hits through cross-polarity matches.")
    if active_demo:
        using_offline_massql = demo_origin == "offline_fallback"
        target_hits, target_massql_frame, target_massql_name = _load_bundled_massql(target_polarity, using_offline_massql)
        source_hits, source_massql_frame, source_massql_name = _load_bundled_massql(source_polarity, using_offline_massql)
        if using_offline_massql:
            st.info("Using small synthetic MassQL hit tables matched to the bundled offline example network. The original task-specific PostMN exports are included below for download and will be used automatically when the real GNPS2 task outputs or a saved snapshot are available.")
        else:
            st.success("Using the task-specific PostMN MassQL Full Table exports supplied with this example.")
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            _show_bundled_massql_preview(target_polarity, target_massql_frame, target_massql_name)
        with ml_col2:
            _show_bundled_massql_preview(source_polarity, source_massql_frame, source_massql_name)
        with st.expander("Download the original task-specific MassQL TSV exports"):
            raw_pos = (EXAMPLES_DIR / "provided_postmn_positive.tsv").read_bytes()
            raw_neg = (EXAMPLES_DIR / "provided_postmn_negative.tsv").read_bytes()
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("Download positive MassQL TSV", raw_pos, file_name="postmn_full_table_pos_both_queries.tsv", mime="text/tab-separated-values")
            with dl2:
                st.download_button("Download negative MassQL TSV", raw_neg, file_name="postmn_full_table_neg.tsv", mime="text/tab-separated-values")
    else:
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            target_hits, target_massql_frame, target_massql_name = _massql_upload_ui(f"{target_polarity.title()}-mode PostMN MassQL Full Table (.tsv)", target_polarity)
        with ml_col2:
            source_hits, source_massql_frame, source_massql_name = _massql_upload_ui(f"{source_polarity.title()}-mode PostMN MassQL Full Table (.tsv)", source_polarity)

    st.divider()
    run_disabled = (
        pos_features is None or neg_features is None or graph is None or graph_bytes is None
        or not pos_adducts or not neg_adducts
        or mass_tolerance_ppm is None or rt_tolerance_min is None
        or (tag_edge_delta_mass and (edge_delta_mass_da is None or edge_delta_mass_tolerance_da is None))
    )
    if not pos_adducts:
        st.warning("Select at least one positive-mode adduct (or add a custom one).")
    if not neg_adducts:
        st.warning("Select at least one negative-mode adduct (or add a custom one).")

    if st.button("Run matching and enrich network", type="primary", disabled=run_disabled):
        try:
            with st.spinner("Calculating neutral masses and matching features…"):
                matches = match_features(
                    positive=pos_features,
                    negative=neg_features,
                    positive_adducts=pos_adducts,
                    negative_adducts=neg_adducts,
                    mass_tolerance_ppm=float(mass_tolerance_ppm),
                    rt_tolerance_min=float(rt_tolerance_min),
                    target_polarity=target_polarity,
                )
                target_features = pos_features if target_polarity == "positive" else neg_features
                source_features = neg_features if target_polarity == "positive" else pos_features
                enriched_graph, summary = enrich_graph(
                    graph=graph,
                    matches=matches,
                    target_features=target_features,
                    source_features=source_features,
                    target_polarity=target_polarity,
                    target_massql_hits=target_hits,
                    source_massql_hits=source_hits,
                )
                enriched_graph, edge_tag_summary = tag_edges_by_delta_mass(
                    graph=enriched_graph,
                    target_features=target_features,
                    enabled=bool(tag_edge_delta_mass),
                    target_delta_mass_da=edge_delta_mass_da if tag_edge_delta_mass else None,
                    tolerance_da=edge_delta_mass_tolerance_da if tag_edge_delta_mass else None,
                )
                summary.update(edge_tag_summary)
                graph_out = graph_to_graphml_bytes(enriched_graph)
                style_out = cytoscape_style_xml()
                summary_out = pd.Series(summary).to_json(indent=2).encode("utf-8")
                match_out = matches.to_csv(index=False).encode("utf-8-sig")
                edge_table = graph_edge_table(enriched_graph)
                edge_table_out = edge_table.to_csv(index=False).encode("utf-8-sig")
                bundle_buffer = io.BytesIO()
                with zipfile.ZipFile(bundle_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.writestr(f"cpne_enriched_{target_polarity}_network.graphml", graph_out)
                    bundle.writestr("cpne_cytoscape_style.xml", style_out)
                    bundle.writestr("cpne_cross_polarity_matches.csv", match_out)
                    bundle.writestr("cpne_network_edge_table.csv", edge_table_out)
                    bundle.writestr("cpne_run_summary.json", summary_out)
                st.session_state["cpne_result"] = {
                    "matches": matches,
                    "summary": summary,
                    "graphml": graph_out,
                    "style_xml": style_out,
                    "summary_json": summary_out,
                    "match_csv": match_out,
                    "edge_table": edge_table,
                    "edge_table_csv": edge_table_out,
                    "bundle_zip": bundle_buffer.getvalue(),
                    "target_polarity": target_polarity,
                    "source_polarity": source_polarity,
                    "source_names": source_names,
                }
        except Exception as exc:
            st.error(f"The matching/enrichment step failed: {exc}")

    result = st.session_state.get("cpne_result")
    if result:
        st.divider()
        st.header("Results")
        summary = result["summary"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Feature pairs matched", f"{summary['matched_feature_pairs']:,}")
        k2.metric("Target features matched", f"{summary['matched_target_features']:,}")
        k3.metric("Target nodes", f"{summary['target_nodes']:,}")
        k4.metric("Target features mapped to graph", f"{summary['mapped_target_features']:,}")
        st.caption(f"Target network: {result['target_polarity']} mode. Source data: {result['source_polarity']} mode. Results reflect the last click on ‘Run matching and enrich network’; rerun after changing inputs or settings.")
        if summary["unmapped_target_features"] > 0:
            st.warning(f"{summary['unmapped_target_features']:,} target table features could not be mapped to GraphML nodes by ID. Their matches are still listed in the table, but those nodes could not be enriched. This often means the node IDs and table feature IDs use different naming conventions.")
        if summary["matched_feature_pairs"] == 0:
            st.warning("No cross-polarity feature pairs met the selected neutral-mass and RT criteria. Check the selected adducts, tolerances and whether both runs used comparable chromatography.")
        if summary.get("edge_delta_mass_tagging_enabled"):
            st.success(
                f"Edge Δm tagging: {summary['edges_matching_delta_mass']:,} of "
                f"{summary['edges_checked_for_delta_mass']:,} edges with precursor m/z values "
                f"matched {summary['edge_target_delta_mass_da']:.6f} ± "
                f"{summary['edge_delta_mass_tolerance_da']:.6f} Da."
            )
            if summary.get("edges_missing_precursor_mz", 0):
                st.warning(f"{summary['edges_missing_precursor_mz']:,} edges could not be evaluated because a connected node had no recoverable precursor m/z.")

        matches = result["matches"]
        st.subheader("Cross-polarity matches")
        if matches.empty:
            st.info("No matches found with the current settings.")
        else:
            st.dataframe(matches, use_container_width=True, hide_index=True)
            st.download_button(
                "Download match table (CSV)",
                data=matches.to_csv(index=False).encode("utf-8-sig"),
                file_name="cpne_cross_polarity_matches.csv",
                mime="text/csv",
            )
        st.subheader("Enriched network")
        st.write("The GraphML includes a plain node-table column **`Matched cross polarities`** with `Yes`/`No`, plus machine-readable `CPNE_` attributes for matches and MassQL hits. Cytoscape may apply its current visual style instead of any appearance suggested by GraphML, so the Yes/No column is the reliable portable flag you can map to border color in Cytoscape. The optional companion style XML can be imported separately, but it may need adjustment for your Cytoscape version/style.")
        if summary.get("edge_delta_mass_tagging_enabled"):
            st.write("The network edge table includes **`Matches selected precursor Δm`**, **`Observed precursor Δm (Da)`**, and **`Precursor Δm error (Da)`**. These are also stored as edge attributes in the GraphML and can be used directly in Cytoscape's Edge Table.")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download enriched GraphML",
                data=result["graphml"],
                file_name=f"cpne_enriched_{result['target_polarity']}_network.graphml",
                mime="application/graphml+xml",
                type="primary",
            )
        with d2:
            st.download_button(
                "Download Cytoscape style XML",
                data=result["style_xml"],
                file_name="cpne_cytoscape_style.xml",
                mime="application/xml",
            )
        with d3:
            st.download_button(
                "Download complete bundle (.zip)",
                data=result["bundle_zip"],
                file_name="cpne_enriched_network_bundle.zip",
                mime="application/zip",
            )
        edge_col1, edge_col2 = st.columns([1, 2])
        with edge_col1:
            st.download_button(
                "Download network edge table (CSV)",
                data=result["edge_table_csv"],
                file_name="cpne_network_edge_table.csv",
                mime="text/csv",
            )
        with edge_col2:
            with st.expander("Preview network edge table"):
                st.dataframe(result["edge_table"], use_container_width=True, hide_index=True)
        st.download_button(
            "Download run summary (JSON)",
            data=result["summary_json"],
            file_name="cpne_run_summary.json",
            mime="application/json",
        )
        with st.expander("Run details"):
            st.json({**summary, "files": result["source_names"], "mass_tolerance_ppm": float(mass_tolerance_ppm), "rt_tolerance_min": float(rt_tolerance_min), "rt_compatibility_required": True, "edge_delta_mass_tagging_enabled": bool(tag_edge_delta_mass), "edge_target_delta_mass_da": edge_delta_mass_da if tag_edge_delta_mass else None, "edge_delta_mass_tolerance_da": edge_delta_mass_tolerance_da if tag_edge_delta_mass else None, "positive_adducts": {k: {"mass_shift_da": v[0], "charge_magnitude": v[1]} for k, v in pos_adducts.items()}, "negative_adducts": {k: {"mass_shift_da": v[0], "charge_magnitude": v[1]} for k, v in neg_adducts.items()}})

    st.divider()
    with st.expander("Important notes / current prototype limits"):
        st.markdown(
            "- The task-ID route downloads **public** GNPS2 task archives; private tasks may not be retrievable without authentication.\n"
            "- Automatic file selection inside task archives is based on known GNPS2 output filenames and is best-effort across workflow versions.\n"
            "- MassQL output columns can differ. Please verify the ID and validation/status columns selected for each upload.\n"
            "- Cross-polarity matches are candidate assignments, not proof that two ions are the same compound. Isomers, in-source fragments, coelution and adduct misassignment can produce false matches.\n"
            "- The most portable Cytoscape annotation is the node-table column `Matched cross polarities` (`Yes`/`No`). GraphML does not reliably force Cytoscape to override the currently selected visual style; map this column to border color in Cytoscape if you want persistent green outlines.\n"
            "- Optional edge tagging compares the absolute precursor m/z difference between the two nodes joined by each existing network edge. It does not search fragment-ion differences or neutral losses. Tagged edge information is stored as GraphML edge attributes and included in `cpne_network_edge_table.csv`."
        )


if __name__ == "__main__":
    main()

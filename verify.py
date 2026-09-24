#!/usr/bin/env python3
"""Executable acceptance suite for the triple-mask assignment workbench.

Runs against a live deployment (default http://localhost:8000) and exits
non-zero if any check fails.  Optimality, canonical normalization,
uniqueness and witness correctness are cross-checked against an exhaustive
brute-force enumerator, so the solver is verified rather than trusted.

Usage:
    python3 verify.py [--base-url http://localhost:8000] [--skip-static]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode())
        except Exception:
            body = None
        return exc.code, body


def get_text(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as resp:
        return resp.status, resp.read().decode()


# --------------------------------------------------------------------------
# Independent brute-force oracle (canonical colorings only)
# --------------------------------------------------------------------------

def brute_force(fragments, conflict_edges, stitch_edges, adjacency_edges=None):
    """Return (best_cost, [canonical optimal color sequences]).

    When ``adjacency_edges`` is supplied, only colorings in which every used
    mask's fragments are mutually reachable through adjacency edges count.
    """
    order = sorted(fragments)
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    conf = [[] for _ in range(n)]
    for a, b in conflict_edges:
        conf[pos[a]].append(pos[b])
        conf[pos[b]].append(pos[a])
    st = [[] for _ in range(n)]
    for a, b, w in stitch_edges:
        st[pos[a]].append((pos[b], w))
        st[pos[b]].append((pos[a], w))
    adj = [[] for _ in range(n)]
    if adjacency_edges is not None:
        for a, b in adjacency_edges:
            adj[pos[a]].append(pos[b])
            adj[pos[b]].append(pos[a])

    def connected(seq, k):
        members = [i for i, c in enumerate(seq) if c == k]
        if len(members) <= 1:
            return True
        seen = {members[0]}
        stack = [members[0]]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if seq[j] == k and j not in seen:
                    seen.add(j)
                    stack.append(j)
        return len(seen) == len(members)

    best = None
    sols = []
    colors = [0] * n

    def rec(i, max_color, cost):
        nonlocal best, sols
        if best is not None and cost > best:
            return
        if i == n:
            if adjacency_edges is not None and not all(
                connected(colors, k) for k in (0, 1, 2)
            ):
                return
            if best is None or cost < best:
                best = cost
                sols = [tuple(colors)]
            elif cost == best:
                sols.append(tuple(colors))
            return
        for k in range(min(max_color + 1, 2) + 1):
            if any(colors[j] == k for j in conf[i] if j < i):
                continue
            extra = sum(w for j, w in st[i] if j < i and colors[j] != k)
            colors[i] = k
            rec(i + 1, max(max_color, k), cost + extra)

    rec(0, -1, 0)
    return best, sols


def is_canonical(seq):
    """First occurrences of colors 0,1,2 must appear in that order."""
    next_color = 0
    for color in seq:
        if color > next_color:
            return False
        if color == next_color:
            next_color += 1
    return True


# --------------------------------------------------------------------------
# Assertions against one optimal response
# --------------------------------------------------------------------------

def check_optimal_response(body, fragments, conflicts, stitches):
    assert body["status"] == "optimal", f"expected optimal, got {body.get('status')}"
    best, sols = brute_force(fragments, conflicts, stitches)
    assert best is not None, "brute force found the instance infeasible"
    order = sorted(fragments)
    pos = {f: i for i, f in enumerate(order)}

    seq = tuple(body["assignment"][str(f)] for f in order)
    assert is_canonical(seq), f"assignment {seq} is not canonical"
    assert body["objective"] == best, (
        f"objective {body['objective']} != brute-force optimum {best}"
    )
    assert seq == min(sols), f"assignment {seq} is not the lexicographic minimum"
    assert body["unique"] == (len(sols) == 1), (
        f"unique={body['unique']} but brute force found {len(sols)} optima"
    )

    expected_cut = sorted(
        (tuple(sorted((a, b))), w)
        for a, b, w in stitches
        if seq[pos[a]] != seq[pos[b]]
    )
    got_cut = sorted(
        (tuple(sorted(edge["pair"])), edge["weight"]) for edge in body["cut_stitches"]
    )
    assert got_cut == expected_cut, f"cut stitches {got_cut} != expected {expected_cut}"
    assert sum(w for _pair, w in got_cut) == best, "cut weights do not sum to objective"

    if len(sols) > 1:
        witness = body.get("witness")
        assert witness is not None, "multiple optima but no witness returned"
        wseq = tuple(witness["assignment"][str(f)] for f in order)
        assert is_canonical(wseq), f"witness {wseq} is not canonical"
        assert wseq in sols, f"witness {wseq} is not an optimal canonical coloring"
        assert wseq != seq, "witness must differ from the primary assignment"
        wcut = sorted(
            (tuple(sorted(e["pair"])), e["weight"]) for e in witness["cut_stitches"]
        )
        assert sum(w for _pair, w in wcut) == best, "witness cut total is not optimal"
    else:
        assert body.get("witness") is None, "unique optimum must not carry a witness"


# --------------------------------------------------------------------------
# Acceptance checks
# --------------------------------------------------------------------------

def check_health(base):
    status, body = request(base, "/api/health")
    assert status == 200, f"health returned HTTP {status}"
    assert body == {"status": "ok"}, f"unexpected health body: {body}"


def check_static_page(base):
    status, text = get_text(base, "/")
    assert status == 200, f"GET / returned HTTP {status}"
    assert 'id="root"' in text, "frontend page does not contain the root element"


def check_unique_optimum(base):
    fragments = [1, 2, 3, 4]
    conflicts = [[1, 2], [2, 3], [1, 3]]
    stitches = [{"pair": [3, 4], "weight": 5}, {"pair": [1, 4], "weight": 1}]
    status, body = request(base, "/api/solve", {
        "fragments": fragments,
        "conflict_edges": conflicts,
        "stitch_edges": stitches,
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body["objective"] == 1, f"objective {body['objective']} != 1"
    assert body["unique"] is True, "expected a unique optimum"
    assert body["assignment"] == {"1": 0, "2": 1, "3": 2, "4": 2}, body["assignment"]
    assert body["cut_stitches"] == [{"pair": [1, 4], "weight": 1}], body["cut_stitches"]
    triples = [(e["pair"][0], e["pair"][1], e["weight"]) for e in stitches]
    check_optimal_response(body, fragments, conflicts, triples)


def check_multiple_optima(base):
    fragments = [1, 2, 3, 4]
    status, body = request(base, "/api/solve", {"fragments": fragments})
    assert status == 200, f"HTTP {status}: {body}"
    assert body["status"] == "optimal"
    assert body["objective"] == 0
    assert body["unique"] is False, "expected multiple optima"
    assert body["assignment"] == {"1": 0, "2": 0, "3": 0, "4": 0}, (
        "lexicographically smallest canonical optimum must be all mask 1"
    )
    assert body["witness"]["assignment"] != body["assignment"]
    check_optimal_response(body, fragments, [], [])


def check_weighted_instance(base):
    fragments = [1, 2, 3, 4, 5, 6]
    conflicts = [[1, 2], [2, 3], [4, 5]]
    stitches = [
        {"pair": [1, 4], "weight": 3},
        {"pair": [2, 5], "weight": 1},
        {"pair": [3, 6], "weight": 2},
        {"pair": [5, 6], "weight": 4},
    ]
    status, body = request(base, "/api/solve", {
        "fragments": fragments,
        "conflict_edges": conflicts,
        "stitch_edges": stitches,
    })
    assert status == 200, f"HTTP {status}: {body}"
    triples = [(e["pair"][0], e["pair"][1], e["weight"]) for e in stitches]
    check_optimal_response(body, fragments, conflicts, triples)


def check_infeasible(base):
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "conflict_edges": [[a, b] for a in range(1, 5) for b in range(a + 1, 5)],
        "stitch_edges": [],
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body["status"] == "infeasible", f"expected infeasible, got {body.get('status')}"


def check_itemized_validation(base):
    status, body = request(base, "/api/solve", {
        "fragments": [1, 1, 2, "x"],
        "conflict_edges": [[1, 9], [2, 2], [1, 2], [1, 2]],
        "stitch_edges": [{"pair": [1, 2], "weight": 3}, {"pair": [2, 3], "weight": 0}],
    })
    assert status == 400, f"expected HTTP 400, got {status}"
    assert body["status"] == "invalid"
    errors = body["errors"]
    assert isinstance(errors, list) and len(errors) >= 8, (
        f"expected at least 8 itemized errors, got {len(errors)}"
    )
    assert all(set(e) == {"loc", "message"} for e in errors)
    locs = {e["loc"] for e in errors}
    for expected in (
        "fragments", "fragments[1]", "fragments[3]",
        "conflict_edges[0]", "conflict_edges[1]", "conflict_edges[3]",
        "stitch_edges[0]", "stitch_edges[1]",
    ):
        assert expected in locs, f"missing error for {expected}"


def check_connectivity_summary(body, fragments, adjacency):
    """Reported adopted edges must connect every used mask's fragments."""
    order = sorted(fragments)
    summary = body.get("connectivity")
    assert summary is not None, "contiguous response must include connectivity"
    assert summary.get("enabled") is True
    adj_set = {tuple(sorted(e)) for e in adjacency}
    by_color = {k: [] for k in (0, 1, 2)}
    for f in order:
        by_color[body["assignment"][str(f)]].append(f)
    covered = []
    for entry in summary["masks"]:
        k = entry["mask"]
        assert entry["fragments"] == sorted(by_color[k]), f"mask {k} coverage mismatch"
        covered.extend(entry["fragments"])
        neigh = {f: set() for f in entry["fragments"]}
        for a, b in entry["adjacency_edges"]:
            assert tuple(sorted((a, b))) in adj_set, "adopted edge not in adjacency list"
            assert body["assignment"][str(a)] == k == body["assignment"][str(b)]
            neigh[a].add(b)
            neigh[b].add(a)
        frags = entry["fragments"]
        if frags:
            seen, stack = {frags[0]}, [frags[0]]
            while stack:
                v = stack.pop()
                for w in neigh[v]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            assert seen == set(frags), f"adopted edges do not connect mask {k}"
    assert sorted(covered) == order, "connectivity summary does not cover all fragments"


def check_contiguous_feasible(base):
    fragments = [1, 2, 3, 4, 5, 6]
    conflicts = [[1, 2], [3, 4]]
    stitches = [{"pair": [2, 5], "weight": 4}, {"pair": [4, 6], "weight": 2}]
    adjacency = [[1, 3], [3, 5], [2, 4], [4, 6]]
    status, body = request(base, "/api/solve", {
        "fragments": fragments,
        "conflict_edges": conflicts,
        "stitch_edges": stitches,
        "contiguous": True,
        "adjacency_edges": adjacency,
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body["status"] == "optimal", body
    triples = [(e["pair"][0], e["pair"][1], e["weight"]) for e in stitches]
    best, sols = brute_force(fragments, conflicts, triples, adjacency)
    assert best is not None, "brute force found the contiguous instance infeasible"
    order = sorted(fragments)
    seq = tuple(body["assignment"][str(f)] for f in order)
    assert seq == min(sols), f"contiguous assignment {seq} not lex-min"
    assert body["objective"] == best, (
        f"contiguous objective {body['objective']} != {best}"
    )
    assert body["unique"] == (len(sols) == 1), (
        f"unique={body['unique']}, brute force found {len(sols)}"
    )
    check_connectivity_summary(body, fragments, adjacency)
    if body.get("witness"):
        check_connectivity_summary(body["witness"], fragments, adjacency)
    else:
        assert len(sols) == 1


def check_contiguous_blocked(base):
    # Triangle 1-2-3 already consumes all three masks; fragment 4's only
    # adjacency edge is 1-4, and 1,4 are forced different by conflict 1-4.
    # The base problem (no connectivity) is feasible.
    fragments = [1, 2, 3, 4]
    conflicts = [[1, 2], [2, 3], [1, 3], [1, 4]]
    adjacency = [[1, 4]]
    status, body = request(base, "/api/solve", {
        "fragments": fragments,
        "conflict_edges": conflicts,
        "contiguous": True,
        "adjacency_edges": adjacency,
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body["status"] == "infeasible", body
    assert body.get("reason") == "connectivity_blocked", body
    # Confirm the base model itself is feasible.
    s0, b0 = request(base, "/api/solve", {"fragments": fragments, "conflict_edges": conflicts})
    assert s0 == 200 and b0["status"] == "optimal", "base instance should be feasible"
    # A genuinely uncolorable graph reports the conflict reason instead.
    s1, b1 = request(base, "/api/solve", {
        "fragments": fragments,
        "conflict_edges": [[a, b] for a in fragments for b in fragments if a < b],
        "contiguous": True,
        "adjacency_edges": adjacency,
    })
    assert s1 == 200 and b1["status"] == "infeasible"
    assert b1.get("reason") == "conflict_graph", b1


def check_contiguous_off_mode_regression(base):
    # Without the flag the request/response shape is the original one.
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "contiguous": False,
        "adjacency_edges": [[1, 9], [2, 2]],  # ignored entirely when off
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body["status"] == "optimal"
    assert "connectivity" not in body and "reason" not in body
    assert set(body) == {
        "status", "objective", "unique", "assignment", "cut_stitches", "witness"
    }
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "conflict_edges": [[a, b] for a in (1, 2, 3, 4) for b in (1, 2, 3, 4) if a < b],
    })
    assert body == {"status": "infeasible"}, "off-mode infeasible body changed"


def check_adjacency_validation(base):
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "contiguous": True,
        "adjacency_edges": [
            [1, 9],
            [2, 2],
            [1, 2],
            [2, 1],
            "nope",
        ],
    })
    assert status == 400, f"expected HTTP 400, got {status}"
    locs = {e["loc"] for e in body["errors"]}
    for expected in (
        "adjacency_edges[0]",  # unknown fragment
        "adjacency_edges[1]",  # self loop
        "adjacency_edges[3]",  # duplicate undirected pair
        "adjacency_edges[4]",  # malformed
    ):
        assert expected in locs, f"missing error for {expected}: {locs}"
    # Empty adjacency with the mode on is invalid.
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "contiguous": True,
        "adjacency_edges": [],
    })
    assert status == 400
    assert any(e["loc"] == "adjacency_edges" for e in body["errors"])


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-static", action="store_true",
                        help="skip the frontend static-page check")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    checks = [
        ("健康检查 /api/health", check_health),
        ("前端页面可由服务访问", check_static_page),
        ("唯一最优解与规范化", check_unique_optimum),
        ("多解时返回字典序最小方案与见证", check_multiple_optima),
        ("带权实例与暴力枚举一致", check_weighted_instance),
        ("无解情形明确区分", check_infeasible),
        ("输入错误逐项返回", check_itemized_validation),
        ("连续掩模岛：可行实例与暴力枚举一致", check_contiguous_feasible),
        ("连续掩模岛：连续性阻断明确说明", check_contiguous_blocked),
        ("连续掩模岛：关闭模式请求/响应回归", check_contiguous_off_mode_regression),
        ("连续掩模岛：邻接输入逐项拒绝", check_adjacency_validation),
    ]

    failures = 0
    for name, fn in checks:
        if fn is check_static_page and args.skip_static:
            print(f"SKIP {name}")
            continue
        try:
            fn(base)
        except Exception as exc:  # noqa: BLE001 - report any failure
            failures += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")

    if failures:
        print(f"\n{failures} 项验收未通过")
        return 1
    print("\n全部验收通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

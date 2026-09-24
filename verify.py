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

def brute_force(fragments, conflict_edges, stitch_edges, adjacency_edges=None,
                contiguous=False):
    """Return (best_cost, [canonical optimal color sequences]).

    With ``contiguous`` set, only colorings whose per-mask fragments are
    mutually reachable over the adjacency edges are admitted.
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
    neigh = [set() for _ in range(n)]
    for a, b in adjacency_edges or []:
        neigh[pos[a]].add(pos[b])
        neigh[pos[b]].add(pos[a])

    def connected(colors):
        for k in range(3):
            nodes = [i for i in range(n) if colors[i] == k]
            if len(nodes) <= 1:
                continue
            seen = {nodes[0]}
            stack = [nodes[0]]
            while stack:
                u = stack.pop()
                for v in neigh[u]:
                    if colors[v] == k and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if set(nodes) - seen:
                return False
        return True

    best = None
    sols = []
    colors = [0] * n

    def rec(i, max_color, cost):
        nonlocal best, sols
        if best is not None and cost > best:
            return
        if i == n:
            if contiguous and not connected(colors):
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


def assert_islands_verify(islands, assignment, adjacency_edges):
    """Per-mask adopted edges must cover exactly the assigned fragments and
    connect them; used so connectivity is reviewed rather than trusted."""
    adj = set(tuple(sorted(e)) for e in adjacency_edges)
    covered = set()
    masks_seen = set()
    for island in islands:
        k = island["mask"]
        assert k in (0, 1, 2), f"bad mask {k} in islands"
        masks_seen.add(k)
        fragments = island["fragments"]
        covered.update(fragments)
        assert sorted(fragments) == sorted(
            int(f) for f, m in assignment.items() if m == k
        ), f"island for mask {k} does not cover exactly its fragments"
        edges = [tuple(sorted(e["pair"])) for e in island["adjacency_edges"]]
        assert all(e in adj for e in edges), "adopted edge not in adjacency list"
        assert all(
            assignment[str(a)] == k and assignment[str(b)] == k for a, b in edges
        ), f"adopted edge on mask {k} joins different-mask endpoints"
        link = {v: set() for v in fragments}
        for a, b in edges:
            link[a].add(b)
            link[b].add(a)
        if fragments:
            seen = {fragments[0]}
            stack = [fragments[0]]
            while stack:
                u = stack.pop()
                for v in link[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            assert seen == set(fragments), (
                f"reported adopted edges do not connect mask {k}: {fragments}"
            )
    assert covered == {int(f) for f in assignment}, "islands miss some fragments"


def check_contiguous_optimal_response(body, fragments, conflicts, stitches, adjacency):
    """Cross-check a contiguous-mode optimum against the brute-force oracle
    (connectivity enforced) and verify every reported island."""
    assert body["status"] == "optimal", f"expected optimal, got {body.get('status')}"
    assert body.get("contiguous") is True, "contiguous flag missing"
    best, sols = brute_force(fragments, conflicts, stitches, adjacency, True)
    assert best is not None, "brute force found no connected coloring"
    order = sorted(fragments)

    seq = tuple(body["assignment"][str(f)] for f in order)
    assert is_canonical(seq), f"assignment {seq} is not canonical"
    assert body["objective"] == best, (
        f"objective {body['objective']} != connected optimum {best}"
    )
    assert seq == min(sols), f"{seq} is not the lexicographic minimum {min(sols)}"
    assert body["unique"] == (len(sols) == 1), (
        f"unique={body['unique']} but oracle found {len(sols)} connected optima"
    )
    assert_islands_verify(body["mask_islands"], body["assignment"], adjacency)
    if body["witness"] is not None:
        assert_islands_verify(
            body["witness"]["mask_islands"], body["witness"]["assignment"], adjacency
        )


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
    assert "reason" not in body, "plain infeasibility must not carry a reason"


def check_contiguous_feasible(base):
    # Plain optimum is 0 (every stitch kept); the connected optimum must cut
    # the weight-4 stitch 2-3 so mask classes stay reachable over adjacency.
    fragments = [1, 2, 3, 4, 5]
    conflicts = []
    stitches = [{"pair": [1, 5], "weight": 5}, {"pair": [2, 3], "weight": 4}]
    adjacency = [[1, 3], [3, 4], [3, 5], [4, 5]]
    status, body = request(base, "/api/solve", {
        "fragments": fragments,
        "conflict_edges": conflicts,
        "stitch_edges": stitches,
        "adjacency_edges": adjacency,
        "contiguous": True,
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body["objective"] == 4, f"expected connected optimum 4, got {body['objective']}"
    triples = [(e["pair"][0], e["pair"][1], e["weight"]) for e in stitches]
    check_contiguous_optimal_response(
        body, fragments, conflicts, triples, adjacency
    )


def check_contiguous_blocked(base):
    # K3 is plainly 3-colorable, but fragment 4 has no adjacency link into
    # the color class it must join: connectivity alone blocks the model.
    payload = {
        "fragments": [1, 2, 3, 4],
        "conflict_edges": [[1, 2], [2, 3], [1, 3]],
        "stitch_edges": [],
        "adjacency_edges": [[1, 2]],
        "contiguous": True,
    }
    status, body = request(base, "/api/solve", payload)
    assert status == 200, f"HTTP {status}: {body}"
    assert body == {"status": "infeasible", "reason": "disconnected"}, body

    # A genuinely non-3-colorable graph in contiguous mode stays a plain
    # infeasibility without the disconnected reason.
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "conflict_edges": [[a, b] for a in range(1, 5) for b in range(a + 1, 5)],
        "adjacency_edges": [[1, 2]],
        "contiguous": True,
    })
    assert status == 200, f"HTTP {status}: {body}"
    assert body == {"status": "infeasible"}, body


def check_contiguous_mode_off_regression(base):
    # Legacy requests (no contiguous fields) keep their exact semantics;
    # sending adjacency data with the switch off must not change anything.
    legacy_payload = {
        "fragments": [1, 2, 3, 4],
        "conflict_edges": [[1, 2], [2, 3], [1, 3]],
        "stitch_edges": [{"pair": [1, 4], "weight": 1}],
    }
    status, legacy = request(base, "/api/solve", legacy_payload)
    assert status == 200, f"HTTP {status}: {legacy}"
    assert "contiguous" not in legacy and "mask_islands" not in legacy

    status, off = request(base, "/api/solve", {
        **legacy_payload,
        "adjacency_edges": [[1, 2]],
        "contiguous": False,
    })
    assert status == 200, f"HTTP {status}: {off}"
    assert off == legacy, f"mode-off response drifted from legacy: {off}"


def check_adjacency_validation(base):
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "contiguous": True,
        "adjacency_edges": [[1, 9], [2, 2], [1, 2], [2, 1]],
    })
    assert status == 400, f"expected HTTP 400, got {status}"
    locs = {e["loc"] for e in body["errors"]}
    assert "adjacency_edges[0]" in locs  # unknown fragment
    assert "adjacency_edges[1]" in locs  # self loop
    assert "adjacency_edges[3]" in locs  # reversed duplicate undirected pair
    assert "adjacency_edges[2]" not in locs
    status, body = request(base, "/api/solve", {
        "fragments": [1, 2, 3, 4],
        "contiguous": "on",
    })
    assert status == 400
    assert "contiguous" in {e["loc"] for e in body["errors"]}


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
        ("连续可行：连通最优与暴力枚举一致", check_contiguous_feasible),
        ("连续性阻断明确标注且与染色无解区分", check_contiguous_blocked),
        ("关闭连续模式的请求/响应回归", check_contiguous_mode_off_regression),
        ("邻接边与开关逐项校验", check_adjacency_validation),
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

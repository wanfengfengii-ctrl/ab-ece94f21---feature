"""Solver correctness tests, including brute-force cross-checks."""

from __future__ import annotations

import random

import pytest

from app.solver import solve_mask_assignment


def brute_force(fragments, conflict_edges, stitch_edges, adjacency_edges=None):
    """Enumerate all canonical colorings; returns (best_cost, solutions).

    When ``adjacency_edges`` is given, only colorings in which every used
    mask's fragments are connected through adjacency edges are admissible.
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
        for k in range(min(max_color + 1, 2) + 1):  # canonical growth only
            if any(colors[j] == k for j in conf[i] if j < i):
                continue
            extra = sum(w for j, w in st[i] if j < i and colors[j] != k)
            colors[i] = k
            rec(i + 1, max(max_color, k), cost + extra)

    rec(0, -1, 0)
    return best, sols


def check_against_brute_force(
    fragments, conflicts, stitches, contiguous=False, adjacency=None
):
    result = solve_mask_assignment(
        fragments,
        conflicts,
        stitches,
        contiguous=contiguous,
        adjacency_edges=adjacency if contiguous else None,
    )
    bf_adj = adjacency if contiguous else None
    best, sols = brute_force(fragments, conflicts, stitches, bf_adj)
    order = sorted(fragments)

    if best is None:
        assert result["status"] == "infeasible"
        if contiguous:
            base_best, _ = brute_force(fragments, conflicts, stitches)
            assert result["reason"] == (
                "connectivity_blocked" if base_best is not None else "conflict_graph"
            )
        return

    assert result["status"] == "optimal"
    assert result["objective"] == best

    seq = tuple(result["assignment"][str(f)] for f in order)
    assert seq == min(sols), "not the lexicographically smallest canonical optimum"
    assert result["unique"] == (len(sols) == 1)

    pos = {f: i for i, f in enumerate(order)}
    expected_cut = sorted(
        (tuple(sorted((a, b))), w) for a, b, w in stitches if seq[pos[a]] != seq[pos[b]]
    )
    got_cut = sorted(
        (tuple(sorted(edge["pair"])), edge["weight"]) for edge in result["cut_stitches"]
    )
    assert got_cut == expected_cut
    assert sum(w for _pair, w in got_cut) == best

    if contiguous:
        assert_connectivity_summary(result, order, adjacency)

    if len(sols) > 1:
        witness = result["witness"]
        assert witness is not None
        wseq = tuple(witness["assignment"][str(f)] for f in order)
        assert wseq in sols
        assert wseq != seq
        if contiguous:
            assert_connectivity_summary(witness, order, adjacency)
    else:
        assert result["witness"] is None


def assert_connectivity_summary(solution, order, adjacency):
    """The reported adopted edges must actually connect each mask island."""
    summary = solution["connectivity"]
    assert summary["enabled"] is True
    adj_set = {tuple(sorted(e)) for e in adjacency}
    reported = set()
    covered = set()
    for entry in summary["masks"]:
        frags = entry["fragments"]
        assert frags == sorted(frags)
        colors = solution["assignment"]
        assert all(colors[str(f)] == entry["mask"] for f in frags)
        covered.update(frags)
        # Re-walk the reported adopted edges.
        neigh = {f: set() for f in frags}
        for a, b in entry["adjacency_edges"]:
            assert tuple(sorted((a, b))) in adj_set, "reported edge is not adjacency"
            reported.add(tuple(sorted((a, b))))
            neigh[a].add(b)
            neigh[b].add(a)
        if frags:
            seen = {frags[0]}
            stack = [frags[0]]
            while stack:
                v = stack.pop()
                for w in neigh[v]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            assert seen == set(frags), f"掩模 {entry['mask']} 的采用边未连通其全部片段"
    assert covered == set(order), "连通性复核未覆盖全部片段"


def test_unique_optimum_with_weighted_stitches():
    fragments = [1, 2, 3, 4]
    conflicts = [[1, 2], [2, 3], [1, 3]]
    stitches = [[3, 4, 5], [1, 4, 1]]
    result = solve_mask_assignment(fragments, conflicts, stitches)
    assert result["status"] == "optimal"
    assert result["objective"] == 1
    assert result["unique"] is True
    assert result["assignment"] == {"1": 0, "2": 1, "3": 2, "4": 2}
    assert result["cut_stitches"] == [{"pair": [1, 4], "weight": 1}]
    assert result["witness"] is None


def test_multiple_optima_returns_lexicographically_smallest_and_witness():
    result = solve_mask_assignment([1, 2, 3, 4], [], [])
    assert result["status"] == "optimal"
    assert result["objective"] == 0
    assert result["unique"] is False
    assert result["assignment"] == {"1": 0, "2": 0, "3": 0, "4": 0}
    witness = result["witness"]
    assert witness is not None
    assert witness["assignment"] != result["assignment"]
    # The witness must itself be canonical (first occurrences 0,1,2 in order).
    seq = [witness["assignment"][str(f)] for f in (1, 2, 3, 4)]
    next_color = 0
    for color in seq:
        assert color <= next_color
        next_color = max(next_color, color + 1)


def test_infeasible_k4():
    conflicts = [[a, b] for a in range(1, 5) for b in range(a + 1, 5)]
    result = solve_mask_assignment([1, 2, 3, 4], conflicts, [])
    assert result["status"] == "infeasible"


def test_non_contiguous_fragment_ids_are_normalized_by_ascending_id():
    fragments = [40, 7, 13, 21]
    conflicts = [[7, 13], [13, 21]]
    stitches = [[7, 40, 2]]
    check_against_brute_force(fragments, conflicts, stitches)


@pytest.mark.parametrize("seed", range(40))
def test_random_instances_match_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 8)
    fragments = sorted(rng.sample(range(1, 60), n))
    pairs = [
        (fragments[i], fragments[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    rng.shuffle(pairs)
    conflicts, stitches = [], []
    for a, b in pairs:
        roll = rng.random()
        if roll < 0.25:
            conflicts.append([a, b])
        elif roll < 0.5:
            stitches.append([a, b, rng.randint(1, 9)])
    check_against_brute_force(fragments, conflicts, stitches)


# --------------------------------------------------------------------------
# Continuous mask islands
# --------------------------------------------------------------------------

def test_contiguous_path_can_share_one_mask():
    fragments = [1, 2, 3, 4]
    adjacency = [[1, 2], [2, 3], [3, 4]]
    result = solve_mask_assignment(
        fragments, [], [], contiguous=True, adjacency_edges=adjacency
    )
    assert result["status"] == "optimal"
    assert result["objective"] == 0
    assert result["assignment"] == {"1": 0, "2": 0, "3": 0, "4": 0}
    summary = result["connectivity"]
    assert summary["enabled"] is True
    assert len(summary["masks"]) == 1
    entry = summary["masks"][0]
    assert entry["mask"] == 0
    assert entry["fragments"] == [1, 2, 3, 4]
    assert entry["adjacency_edges"] == [[1, 2], [2, 3], [3, 4]]
    # A connected partition with a singleton second mask is also optimal.
    assert result["unique"] is False
    witness = result["witness"]
    assert witness is not None
    assert witness["connectivity"]["enabled"] is True


def test_contiguous_optimum_cross_checked():
    # Adjacency only links 1-3 and 2-4; conflict 1-2 then forces the paired
    # layout [0,1,0,1] (up to singleton-color variants), which cuts stitch
    # 3-4.  Connectivity is part of the exact solve, not a post-filter.
    fragments = [1, 2, 3, 4]
    conflicts = [[1, 2]]
    stitches = [[3, 4, 1]]
    adjacency = [[1, 3], [2, 4]]
    check_against_brute_force(fragments, conflicts, stitches, True, adjacency)
    result = solve_mask_assignment(
        fragments, conflicts, stitches, contiguous=True, adjacency_edges=adjacency
    )
    assert result["objective"] == 1
    assert result["assignment"]["1"] != result["assignment"]["2"]
    assert result["assignment"]["1"] == result["assignment"]["3"]
    assert result["assignment"]["2"] == result["assignment"]["4"]


def test_connectivity_blocked_is_distinguished():
    # Triangle 1-2-3 already uses all three masks; fragment 4 has no
    # adjacency edge at all, so it cannot join any mask island.  The base
    # problem (without connectivity) is perfectly feasible.
    fragments = [1, 2, 3, 4]
    conflicts = [[1, 2], [2, 3], [1, 3]]
    result = solve_mask_assignment(
        fragments,
        conflicts,
        [],
        contiguous=True,
        adjacency_edges=[[1, 2]],
    )
    assert result["status"] == "infeasible"
    assert result["reason"] == "connectivity_blocked"


def test_conflict_infeasible_remains_conflict_graph_reason():
    conflicts = [[a, b] for a in range(1, 5) for b in range(a + 1, 5)]
    result = solve_mask_assignment(
        [1, 2, 3, 4],
        conflicts,
        [],
        contiguous=True,
        adjacency_edges=[[1, 2], [2, 3], [3, 4]],
    )
    assert result["status"] == "infeasible"
    assert result["reason"] == "conflict_graph"


def test_off_mode_response_shape_is_unchanged():
    result = solve_mask_assignment([1, 2, 3, 4], [], [])
    assert "connectivity" not in result
    assert set(result) == {
        "status", "objective", "unique", "assignment", "cut_stitches", "witness"
    }
    conflicts = [[a, b] for a in range(1, 5) for b in range(a + 1, 5)]
    infeasible = solve_mask_assignment([1, 2, 3, 4], conflicts, [])
    assert infeasible == {"status": "infeasible"}


@pytest.mark.parametrize("seed", range(30))
def test_random_contiguous_instances_match_brute_force(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(4, 7)
    fragments = sorted(rng.sample(range(1, 60), n))
    pairs = [
        (fragments[i], fragments[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    rng.shuffle(pairs)
    conflicts, stitches, adjacency = [], [], []
    for a, b in pairs:
        roll = rng.random()
        if roll < 0.2:
            conflicts.append([a, b])
        elif roll < 0.4:
            stitches.append([a, b, rng.randint(1, 9)])
        elif roll < 0.65:
            adjacency.append([a, b])
    # Guarantee at least one usable edge (validation requires this too).
    if not adjacency:
        adjacency.append([fragments[0], fragments[1]])
    check_against_brute_force(
        fragments, conflicts, stitches, contiguous=True, adjacency=adjacency
    )

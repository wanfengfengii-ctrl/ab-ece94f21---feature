"""Solver correctness tests, including brute-force cross-checks."""

from __future__ import annotations

import random

import pytest

from app.solver import solve_mask_assignment


def brute_force(fragments, conflict_edges, stitch_edges, adjacency_edges=None,
                contiguous=False):
    """Enumerate all canonical colorings; returns (best_cost, solutions).

    When ``contiguous`` is set, only colorings whose per-mask fragments form
    a single connected component over the adjacency edges are admitted.
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
        for k in range(min(max_color + 1, 2) + 1):  # canonical growth only
            if any(colors[j] == k for j in conf[i] if j < i):
                continue
            extra = sum(w for j, w in st[i] if j < i and colors[j] != k)
            colors[i] = k
            rec(i + 1, max(max_color, k), cost + extra)

    rec(0, -1, 0)
    return best, sols


def islands_are_connected(result, adjacency_edges, assignment_field="assignment",
                          islands_field="mask_islands"):
    """Independently verify per-mask coverage and connectivity of islands."""
    adj = {}
    for a, b in adjacency_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    assignment = result[assignment_field]
    covered = set()
    for island in result[islands_field]:
        k = island["mask"]
        fragments = island["fragments"]
        covered.update(fragments)
        # Coverage: exactly the fragments assigned to this mask.
        assert sorted(int(f) for f, m in assignment.items() if m == k) == sorted(
            fragments
        )
        # Adopted edges must join same-mask endpoints.
        edge_set = {tuple(sorted(e["pair"])) for e in island["adjacency_edges"]}
        assert edge_set <= {
            tuple(sorted((a, b))) for a, b in adjacency_edges
            if assignment[str(a)] == k and assignment[str(b)] == k
        }
        # Connectivity over the reported adopted edges.
        link = {v: set() for v in fragments}
        for a, b in edge_set:
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
            assert seen == set(fragments), f"mask {k} island not connected: {fragments}"
    assert covered == {int(f) for f in assignment}


def check_against_brute_force(fragments, conflicts, stitches, adjacency=None,
                              contiguous=False):
    result = solve_mask_assignment(
        fragments, conflicts, stitches, adjacency, contiguous
    )
    best, sols = brute_force(
        fragments, conflicts, stitches, adjacency, contiguous
    )
    order = sorted(fragments)

    if best is None:
        assert result["status"] == "infeasible"
        if contiguous:
            plain_best, _ = brute_force(fragments, conflicts, stitches)
            if plain_best is not None:
                assert result.get("reason") == "disconnected"
            else:
                assert "reason" not in result
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
        assert result["contiguous"] is True
        islands_are_connected(result, adjacency)

    if len(sols) > 1:
        witness = result["witness"]
        assert witness is not None
        wseq = tuple(witness["assignment"][str(f)] for f in order)
        assert wseq in sols
        assert wseq != seq
        if contiguous:
            islands_are_connected(witness, adjacency)
    else:
        assert result["witness"] is None


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


CONTIG_FEASIBLE = dict(
    fragments=[1, 2, 3, 4, 5],
    conflicts=[],
    stitches=[[1, 5, 5], [2, 3, 4]],
    adjacency=[[1, 3], [3, 4], [3, 5], [4, 5]],
)


def test_contiguous_feasible_matches_brute_force():
    check_against_brute_force(
        CONTIG_FEASIBLE["fragments"],
        CONTIG_FEASIBLE["conflicts"],
        CONTIG_FEASIBLE["stitches"],
        CONTIG_FEASIBLE["adjacency"],
        contiguous=True,
    )


def test_contiguous_mode_can_raise_the_optimum():
    # Without connectivity every stitch can be kept (objective 0); the
    # connected optimum must cut the 2-3 stitch (objective 4) to keep mask
    # classes connected.
    plain = solve_mask_assignment(
        CONTIG_FEASIBLE["fragments"],
        CONTIG_FEASIBLE["conflicts"],
        CONTIG_FEASIBLE["stitches"],
    )
    assert plain["objective"] == 0

    result = solve_mask_assignment(
        CONTIG_FEASIBLE["fragments"],
        CONTIG_FEASIBLE["conflicts"],
        CONTIG_FEASIBLE["stitches"],
        CONTIG_FEASIBLE["adjacency"],
        True,
    )
    assert result["status"] == "optimal"
    assert result["objective"] == 4
    assert result["contiguous"] is True
    islands_are_connected(result, CONTIG_FEASIBLE["adjacency"])
    if result["witness"] is not None:
        islands_are_connected(
            result["witness"], CONTIG_FEASIBLE["adjacency"]
        )


def test_contiguous_blockage_is_distinguished_from_plain_infeasibility():
    # K3 forces three colors; fragment 4 has no adjacency edge into the
    # color class it must join, so connectivity alone blocks the model.
    conflicts = [[1, 2], [2, 3], [1, 3]]
    plain = solve_mask_assignment([1, 2, 3, 4], conflicts, [])
    assert plain["status"] == "optimal"

    blocked = solve_mask_assignment(
        [1, 2, 3, 4], conflicts, [], [[1, 2]], True
    )
    assert blocked == {"status": "infeasible", "reason": "disconnected"}


def test_plain_infeasibility_has_no_disconnected_reason_in_contiguous_mode():
    conflicts = [[a, b] for a in range(1, 5) for b in range(a + 1, 5)]
    result = solve_mask_assignment([1, 2, 3, 4], conflicts, [], [[1, 2]], True)
    assert result == {"status": "infeasible"}


def test_contiguous_without_adjacency_blocks_multiple_fragments_per_mask():
    # Four fragments, three masks, no adjacency edges: every coloring puts
    # at least two fragments together with no edge between them.
    result = solve_mask_assignment([1, 2, 3, 4], [], [], [], True)
    assert result == {"status": "infeasible", "reason": "disconnected"}


def test_contiguous_multi_optima_witness_is_connected():
    result = solve_mask_assignment(
        [1, 2, 3, 4, 5, 6],
        [[1, 2], [2, 5]],
        [[1, 4, 6]],
        [[2, 4], [2, 6], [3, 4], [4, 5], [5, 6]],
        True,
    )
    assert result["status"] == "optimal"
    assert result["objective"] == 6
    assert result["unique"] is False
    islands_are_connected(result, [[2, 4], [2, 6], [3, 4], [4, 5], [5, 6]])
    witness = result["witness"]
    assert witness is not None
    assert witness["assignment"] != result["assignment"]
    islands_are_connected(witness, [[2, 4], [2, 6], [3, 4], [4, 5], [5, 6]])


def test_mode_off_keeps_legacy_response_shape():
    result = solve_mask_assignment(
        [1, 2, 3, 4], [], [], [[1, 2]], contiguous=False
    )
    assert "contiguous" not in result
    assert "mask_islands" not in result
    # Default argument is also off.
    default = solve_mask_assignment([1, 2, 3, 4], [], [], [[1, 2]])
    assert "mask_islands" not in default


@pytest.mark.parametrize("seed", range(30))
def test_random_contiguous_instances_match_brute_force(seed):
    rng = random.Random(100 + seed)
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
        elif roll < 0.38:
            stitches.append([a, b, rng.randint(1, 9)])
        elif roll < 0.6:
            adjacency.append([a, b])
    check_against_brute_force(
        fragments, conflicts, stitches, adjacency, contiguous=True
    )

"""Exact three-mask assignment with stitch minimization.

Every layout fragment is assigned to one of three masks so that each
conflict edge is bichromatic, while the total weight of stitch edges whose
endpoints land on different masks ("cut" stitches) is minimized.

All reported colorings are canonical: scanning fragments in ascending id
order, the first mask encountered is 0, the next new mask is 1, then 2.
Canonical form quotients out the six mask permutations, which makes
uniqueness of the optimum well defined.

With ``contiguous=True`` the "continuous mask island" requirement is added
to the very same MILP: for every mask actually used, all fragments assigned
to it must be mutually reachable using only the supplied undirected
adjacency edges whose endpoints share that mask.  Connectivity is modelled
exactly with one rooted single-commodity flow per used mask: each non-root
member consumes one unit, the mask's first fragment emits the rest, and
flow only crosses same-mask adjacency edges.  Hence the connectivity
requirement, conflict bichromacy, stitch minimization, canonicalization
and uniqueness probing all hold in one exact solve — solutions are never
computed first and filtered afterwards.
"""

from __future__ import annotations

import pulp

MASKS = (0, 1, 2)
TIME_LIMIT_SECONDS = 30


class SolverError(Exception):
    """The solver could not certify an optimal solution."""


def _build_problem(order, conflict_edges, stitch_edges, contiguous=False, adjacency_edges=()):
    """Build the canonical-form MILP for one instance.

    When ``contiguous`` is true, connectivity variables/constraints are
    added and the returned ``aux`` dict carries the handles needed to read
    off which adjacency edges carry each mask's connectivity flow.
    """
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    prob = pulp.LpProblem("mask_assignment", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(n), MASKS), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", range(len(stitch_edges)), lowBound=0, upBound=1)

    for i in range(n):
        prob += pulp.lpSum(x[i][k] for k in MASKS) == 1, f"assign_{i}"

    # Canonical first-occurrence order: fragment i may take mask k > 0 only
    # if some earlier fragment (ascending id) already took mask k - 1.
    for i in range(n):
        for k in (1, 2):
            prob += (
                x[i][k] <= pulp.lpSum(x[j][k - 1] for j in range(i)),
                f"canon_{i}_{k}",
            )

    for a, b in conflict_edges:
        ia, ib = pos[a], pos[b]
        for k in MASKS:
            prob += x[ia][k] + x[ib][k] <= 1, f"conflict_{ia}_{ib}_{k}"

    for ei, (a, b, _w) in enumerate(stitch_edges):
        ia, ib = pos[a], pos[b]
        for k in MASKS:
            prob += y[ei] >= x[ia][k] - x[ib][k], f"cut_{ei}_{k}"

    objective = pulp.lpSum(w * y[ei] for ei, (_a, _b, w) in enumerate(stitch_edges))
    prob += objective

    aux = {}
    if not contiguous:
        return prob, x, objective, aux

    # --- Continuous mask islands -----------------------------------------
    # One single-commodity flow per used mask k, rooted at k's first
    # fragment.  Every non-root member consumes one unit, the root emits
    # count(k)-1 units, and flow may only traverse edges whose two
    # endpoints both carry k.  Hence a feasible flow exists exactly when
    # k's members are mutually reachable through adjacency edges.
    #
    # All added variables are continuous: flow is bounded by the existing
    # binary color variables, so CBC still branches only on the coloring.
    # Once colors are integral the flow subproblem is a pure (totally
    # unimodular) network LP.
    adj = [(pos[a], pos[b]) for a, b in adjacency_edges]
    count = {k: pulp.lpSum(x[i][k] for i in range(n)) for k in MASKS}
    # root[i,k] marks k's first (ascending id) member.  Fully determined by
    # the colors, so it adds essentially no branching; using it keeps the
    # supply linearization small instead of pairwise O(n^2) constraints.
    root = pulp.LpVariable.dicts("root", (range(n), MASKS), cat=pulp.LpBinary)
    supply = pulp.LpVariable.dicts("supply", (range(n), MASKS), lowBound=0, upBound=n)
    flow = {}  # (edge, tail, head, mask) -> nonnegative flow

    incidence = [[] for _ in range(n)]
    for ei, (ia, ib) in enumerate(adj):
        incidence[ia].append((ei, ib))
        incidence[ib].append((ei, ia))
        for k in MASKS:
            fwd = pulp.LpVariable(f"flow_{ei}_{ia}_{ib}_{k}", lowBound=0, upBound=n - 1)
            rev = pulp.LpVariable(f"flow_{ei}_{ib}_{ia}_{k}", lowBound=0, upBound=n - 1)
            flow[(ei, ia, ib, k)] = fwd
            flow[(ei, ib, ia, k)] = rev
            # Flow only crosses an edge while both endpoints carry mask k.
            prob += (
                fwd + rev <= (n - 1) * x[ia][k],
                f"arc_at_a_{ei}_{k}",
            )
            prob += (
                fwd + rev <= (n - 1) * x[ib][k],
                f"arc_at_b_{ei}_{k}",
            )

    for k in MASKS:
        prob += pulp.lpSum(root[i][k] for i in range(n)) <= 1, f"one_root_{k}"

    for i in range(n):
        for k in MASKS:
            prob += root[i][k] <= x[i][k], f"root_member_{i}_{k}"
            # The first member must be the (unique) root.
            prob += (
                root[i][k] >= x[i][k] - pulp.lpSum(x[j][k] for j in range(i)),
                f"root_first_{i}_{k}",
            )
            # supply[i,k] = count(k) at the root, 0 elsewhere.  Continuous:
            # fixed once the binary colors/roots are fixed.
            prob += supply[i][k] <= count[k], f"supply_le_count_{i}_{k}"
            prob += supply[i][k] <= n * root[i][k], f"supply_le_root_{i}_{k}"
            prob += (
                supply[i][k] >= count[k] - n * (1 - root[i][k]),
                f"supply_eq_{i}_{k}",
            )
            # The root (supply = count) emits count - 1; every other member
            # has net inflow 1; non-members carry no flow at all.
            net = pulp.lpSum(
                flow[(ei, other, i, k)] - flow[(ei, i, other, k)]
                for ei, other in incidence[i]
            )
            prob += net == x[i][k] - supply[i][k], f"flow_balance_{i}_{k}"

    aux["x"] = x
    aux["y"] = y
    aux["root"] = root
    aux["supply"] = supply
    aux["flow"] = flow
    aux["adjacency_edges"] = adjacency_edges
    return prob, x, objective, aux


def _cbc(warm_start=False):
    return pulp.PULP_CBC_CMD(
        msg=False, timeLimit=TIME_LIMIT_SECONDS, warmStart=warm_start
    )


def _status(prob):
    return pulp.LpStatus[prob.status]


def _color_of(x, i):
    return max(MASKS, key=lambda k: pulp.value(x[i][k]) or 0.0)


def _apply_warm_start(aux, order, pos, colors, stitches):
    """Seed every MILP variable with a fully feasible assignment.

    CBC only accepts a MIP start when it satisfies all constraints; because
    pinned colors change between lex-min solves, the continuous flow values
    from the previous solve would be stale.  We therefore rebuild a valid
    rooted BFS-tree flow for each mask from the given coloring and seed
    every binary and continuous variable ourselves.
    """
    n = len(order)
    x, y = aux["x"], aux["y"]
    root, supply, flow = aux["root"], aux["supply"], aux["flow"]
    adjacency = aux["adjacency_edges"]

    for i in range(n):
        for k in MASKS:
            x[i][k].setInitialValue(1 if colors[i] == k else 0)
    for ei, (a, b, _w) in enumerate(stitches):
        y[ei].setInitialValue(1 if colors[pos[a]] != colors[pos[b]] else 0)

    adj_index = [[] for _ in range(n)]
    for ei, (a, b) in enumerate(adjacency):
        adj_index[pos[a]].append((ei, pos[b]))
        adj_index[pos[b]].append((ei, pos[a]))

    # Default everything to zero, then lay down tree flows.
    for i in range(n):
        for k in MASKS:
            root[i][k].setInitialValue(0)
            supply[i][k].setInitialValue(0)
    for (ei, ia, ib, k), var in flow.items():
        var.setInitialValue(0)

    for k in MASKS:
        members = [i for i in range(n) if colors[i] == k]
        if not members:
            continue
        root_idx = members[0]
        root[root_idx][k].setInitialValue(1)
        # BFS tree from the root along same-color adjacency edges.
        parent = {root_idx: None}
        parent_edge = {}
        queue = [root_idx]
        for v in queue:
            for ei, w in adj_index[v]:
                if w not in parent and colors[w] == k:
                    parent[w] = v
                    parent_edge[w] = ei
                    queue.append(w)
        # Post-order subtree sizes; flow on edge v->w carries subtree(w).
        subtree = {v: 1 for v in members}
        for v in reversed(queue):
            if parent[v] is not None:
                subtree[parent[v]] += subtree[v]
                ei = parent_edge[v]
                flow[(ei, parent[v], v, k)].setInitialValue(subtree[v])
        for v in members:
            supply[v][k].setInitialValue(len(members) if v == root_idx else 0)


def _cut_stitches(stitch_edges, pos, colors):
    return [
        {"pair": [a, b], "weight": w}
        for a, b, w in stitch_edges
        if colors[pos[a]] != colors[pos[b]]
    ]


def _connectivity_summary(order, pos, colors, adjacency_edges, aux):
    """Per-mask fragments and the adjacency edges carrying connectivity flow.

    The returned edges are re-checked here with an explicit reachability
    walk, so a numerical slip could never report a false "connected".
    """
    members = {k: set() for k in MASKS}
    for i, v in enumerate(order):
        members[colors[i]].add(v)

    flow = aux["flow"]
    used = {k: [] for k in MASKS}
    # A flow-bearing arc marks its edge as part of the connecting network.
    for ei, (a, b) in enumerate(adjacency_edges):
        ia, ib = pos[a], pos[b]
        for k in MASKS:
            fwd = pulp.value(flow[(ei, ia, ib, k)]) or 0.0
            rev = pulp.value(flow[(ei, ib, ia, k)]) or 0.0
            if fwd > 0.5 or rev > 0.5:
                if colors[pos[a]] != k or colors[pos[b]] != k:
                    raise SolverError("求解结果中的连通边跨越了不同掩模")
                pair = [a, b] if a < b else [b, a]
                if pair not in used[k]:
                    used[k].append(pair)

    masks = []
    for k in MASKS:
        frags = sorted(members[k])
        if not frags:
            continue
        edges = sorted(used[k])
        adj = {v: set() for v in frags}
        for a, b in edges:
            if a not in adj or b not in adj:
                raise SolverError("求解结果采用了不属于该掩模的邻接边")
            adj[a].add(b)
            adj[b].add(a)
        seen = {frags[0]}
        stack = [frags[0]]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if seen != set(frags):
            raise SolverError("求解结果未通过掩模连通性复核")
        masks.append(
            {
                "mask": k,
                "fragments": frags,
                "adjacency_edges": edges,
            }
        )
    return {"enabled": True, "masks": masks}


def solve_mask_assignment(
    fragments,
    conflict_edges,
    stitch_edges,
    contiguous=False,
    adjacency_edges=None,
):
    """Solve one validated instance.

    Returns ``{"status": "infeasible"}`` when the conflict graph admits no
    three-mask coloring.  Otherwise returns the lexicographically smallest
    canonical optimum (fragment ids ascending), whether that optimum is the
    unique canonical optimum, and — when it is not — a second, different
    canonical optimum as a witness.

    With ``contiguous=True`` every solution (primary and witness) must keep
    each mask's fragments connected through ``adjacency_edges``; the
    infeasible response then carries ``reason`` set to
    ``"connectivity_blocked"`` when the base model is feasible on its own,
    or ``"conflict_graph"`` when the conflict graph itself is not
    3-colorable.
    """
    order = sorted(fragments)
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    stitches = [tuple(edge) for edge in stitch_edges]
    adjacency = [tuple(edge) for edge in (adjacency_edges or [])]

    prob, x, objective, aux = _build_problem(
        order, conflict_edges, stitches, contiguous, adjacency
    )
    prob.solve(_cbc())
    status = _status(prob)
    if status == "Infeasible":
        if not contiguous:
            return {"status": "infeasible"}
        # Classify the infeasibility: is the base model itself feasible?
        base, _bx, _bo, _baux = _build_problem(order, conflict_edges, stitches)
        base.solve(_cbc())
        base_status = _status(base)
        if base_status == "Infeasible":
            return {"status": "infeasible", "reason": "conflict_graph"}
        if base_status != "Optimal":
            raise SolverError(f"求解器未能判定无救原因（状态：{base_status}）")
        return {"status": "infeasible", "reason": "connectivity_blocked"}
    if status != "Optimal":
        raise SolverError(f"求解器未能求得最优解（状态：{status}）")

    # Read the optimum off the optimal coloring itself: stitch weights are
    # positive integers, so the cut total is an exact integer.
    first_colors = [_color_of(x, i) for i in range(n)]
    best = sum(w for a, b, w in stitches if first_colors[pos[a]] != first_colors[pos[b]])
    prob += objective <= best + 1e-6, "optimal_bound"

    # Lexicographically smallest canonical optimum.  Instead of one CBC
    # launch per fragment (process/file overhead dominates), minimize the
    # colors of a window encoded as one base-3 integer: a difference at an
    # earlier position outweighs every later combination, so one solve
    # determines the whole window's lex-min colors.  Windows are pinned in
    # turn, ids ascending.
    WINDOW = 8
    colors = [0] * n
    current = list(first_colors)
    lo = 0
    while lo < n:
        hi = min(lo + WINDOW, n)
        window_obj = pulp.lpSum(
            3 ** (hi - 1 - i) * pulp.lpSum(k * x[i][k] for k in MASKS)
            for i in range(lo, hi)
        )
        if contiguous:
            _apply_warm_start(aux, order, pos, current, stitches)
        prob.setObjective(window_obj)
        prob.solve(_cbc(warm_start=contiguous))
        if _status(prob) != "Optimal":
            raise SolverError("求解器在构造字典序最小方案时失败")
        for i in range(lo, hi):
            colors[i] = _color_of(x, i)
            prob += x[i][colors[i]] == 1, f"fix_{i}"
        current = [_color_of(x, j) for j in range(n)]
        lo = hi

    connectivity = None
    if contiguous:
        connectivity = _connectivity_summary(
            order, pos, colors, adjacency, aux
        )

    # Uniqueness probe: any canonical optimum different from the one above?
    for i in range(n):
        prob.constraints.pop(f"fix_{i}", None)
    prob += pulp.lpSum(x[i][colors[i]] for i in range(n)) <= n - 1, "exclude_assignment"
    prob.solve(_cbc())
    unique = _status(prob) != "Optimal"

    witness = None
    if not unique:
        witness_colors = [_color_of(x, i) for i in range(n)]
        witness = {
            "assignment": {str(order[i]): witness_colors[i] for i in range(n)},
            "cut_stitches": _cut_stitches(stitches, pos, witness_colors),
        }
        if contiguous:
            witness["connectivity"] = _connectivity_summary(
                order, pos, witness_colors, adjacency, aux
            )

    result = {
        "status": "optimal",
        "objective": best,
        "unique": unique,
        "assignment": {str(order[i]): colors[i] for i in range(n)},
        "cut_stitches": _cut_stitches(stitches, pos, colors),
        "witness": witness,
    }
    if contiguous:
        result["connectivity"] = connectivity
    return result

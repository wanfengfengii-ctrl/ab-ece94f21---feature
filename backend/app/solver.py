"""Exact three-mask assignment with stitch minimization.

Every layout fragment is assigned to one of three masks so that each
conflict edge is bichromatic, while the total weight of stitch edges whose
endpoints land on different masks ("cut" stitches) is minimized.

All reported colorings are canonical: scanning fragments in ascending id
order, the first mask encountered is 0, the next new mask is 1, then 2.
Canonical form quotients out the six mask permutations, which makes
uniqueness of the optimum well defined.

Optional "contiguous mask islands" mode (``contiguous=True``): for every
mask actually used, all fragments assigned to it must be mutually reachable
using only the supplied undirected adjacency edges whose endpoints share
that mask.  The connectivity requirement is part of the same MILP — it is
never enforced by solving first and filtering afterwards — so optimality,
canonical normalization and uniqueness all hold over the connected
feasible region.  If the plain model is feasible but this model is not,
the result reports ``reason: "disconnected"``.
"""

from __future__ import annotations

import pulp

MASKS = (0, 1, 2)
TIME_LIMIT_SECONDS = 30


class SolverError(Exception):
    """The solver could not certify an optimal solution."""


def _build_problem(order, conflict_edges, stitch_edges, adjacency_edges, contiguous):
    """Build the canonical-form MILP for one instance."""
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

    # Contiguous mask islands: for each mask k, the fragments colored k must
    # induce one connected component over the given undirected adjacency
    # edges.  Modeled as a single-commodity directed flow per mask rooted
    # at the lowest-id fragment colored k: the root ships one unit to every
    # other fragment colored k, and flow may only traverse edges whose
    # both ends share color k.  A mask with zero or one fragment is
    # trivially connected.
    if contiguous:
        out_adj = [[] for _ in range(n)]
        in_adj = [[] for _ in range(n)]
        for ei, (a, b) in enumerate(adjacency_edges):
            ia, ib = pos[a], pos[b]
            # fp: ia -> ib, fn: ib -> ia (one variable per direction).
            out_adj[ia].append((ib, ei, "p"))
            in_adj[ib].append((ia, ei, "p"))
            out_adj[ib].append((ia, ei, "n"))
            in_adj[ia].append((ib, ei, "n"))
        for k in MASKS:
            fp = pulp.LpVariable.dicts(
                f"fp_{k}", range(len(adjacency_edges)), lowBound=0, upBound=n
            )
            fn = pulp.LpVariable.dicts(
                f"fn_{k}", range(len(adjacency_edges)), lowBound=0, upBound=n
            )
            flow = {"p": fp, "n": fn}
            # Flow only traverses an edge when both endpoints take mask k.
            for ei, (a, b) in enumerate(adjacency_edges):
                ia, ib = pos[a], pos[b]
                for d, var in (("p", fp[ei]), ("n", fn[ei])):
                    prob += var <= n * x[ia][k], f"flowcap_{k}_{ei}_{d}_{ia}"
                    prob += var <= n * x[ib][k], f"flowcap_{k}_{ei}_{d}_{ib}"

            # The bounds below pin r[i] to 1 exactly when i is the lowest-id
            # fragment colored k (its indicator needs no binary declaration).
            r = pulp.LpVariable.dicts(f"root_{k}", range(n), lowBound=0, upBound=1)
            z = pulp.LpVariable.dicts(f"rootsup_{k}", range(n), lowBound=0, upBound=n)
            total_k = pulp.lpSum(x[j][k] for j in range(n))
            for i in range(n):
                prob += r[i] <= x[i][k], f"rootcol_{k}_{i}"
                for j in range(i):
                    prob += r[i] <= 1 - x[j][k], f"rootfirst_{k}_{i}_{j}"
                prob += (
                    r[i] >= x[i][k] - pulp.lpSum(x[j][k] for j in range(i)),
                    f"rootforce_{k}_{i}",
                )
                # z[i] = total_k * r[i], linearized.
                prob += z[i] <= n * r[i], f"rootsup_a_{k}_{i}"
                prob += z[i] <= total_k, f"rootsup_b_{k}_{i}"
                prob += z[i] >= total_k - n * (1 - r[i]), f"rootsup_c_{k}_{i}"
                # Root emits size-1, every other colored node consumes 1.
                outflow = pulp.lpSum(flow[d][ei] for _nb, ei, d in out_adj[i])
                inflow = pulp.lpSum(flow[d][ei] for _nb, ei, d in in_adj[i])
                prob += outflow - inflow == z[i] - x[i][k], f"flowbal_{k}_{i}"

    objective = pulp.lpSum(w * y[ei] for ei, (_a, _b, w) in enumerate(stitch_edges))
    prob += objective
    return prob, x, objective


def _cbc():
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=TIME_LIMIT_SECONDS)


def _status(prob):
    return pulp.LpStatus[prob.status]


def _color_of(x, i):
    return max(MASKS, key=lambda k: pulp.value(x[i][k]) or 0.0)


def _cut_stitches(stitch_edges, pos, colors):
    return [
        {"pair": [a, b], "weight": w}
        for a, b, w in stitch_edges
        if colors[pos[a]] != colors[pos[b]]
    ]


def _adopted_adjacency(adjacency_edges, pos, colors):
    """Adjacency edges actually used as connectivity witnesses.

    An edge is adopted when both endpoints are assigned to the same mask;
    only such edges can carry intra-mask reachability.  Returned grouped by
    mask with the fragments covered on that mask, so a reviewer can verify
    each island's connectivity directly.
    """
    groups = {k: {"edges": [], "fragments": []} for k in MASKS}
    for a, b in adjacency_edges:
        ca, cb = colors[pos[a]], colors[pos[b]]
        if ca == cb:
            groups[ca]["edges"].append([a, b])
    for v, i in pos.items():
        groups[colors[i]]["fragments"].append(v)
    islands = []
    for k in MASKS:
        fragments = sorted(groups[k]["fragments"])
        if not fragments:
            continue
        islands.append(
            {
                "mask": k,
                "fragments": fragments,
                "adjacency_edges": [
                    {"pair": pair} for pair in sorted(map(tuple, groups[k]["edges"]))
                ],
            }
        )
    return islands


def solve_mask_assignment(
    fragments,
    conflict_edges,
    stitch_edges,
    adjacency_edges=None,
    contiguous=False,
):
    """Solve one validated instance.

    Returns ``{"status": "infeasible"}`` when the conflict graph admits no
    three-mask coloring.  In contiguous mode, plain-coloring infeasibility
    and connectivity-only infeasibility are distinguished: the latter
    returns ``{"status": "infeasible", "reason": "disconnected"}``.
    Otherwise returns the lexicographically smallest canonical optimum
    (fragment ids ascending), whether that optimum is the unique canonical
    optimum, and — when it is not — a second, different canonical optimum
    as a witness.
    """
    order = sorted(fragments)
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    stitches = [tuple(edge) for edge in stitch_edges]
    adjacency = [tuple(edge) for edge in (adjacency_edges or [])]

    prob, x, objective = _build_problem(
        order, conflict_edges, stitches, adjacency, contiguous
    )
    prob.solve(_cbc())
    status = _status(prob)
    if status == "Infeasible":
        result = {"status": "infeasible"}
        if contiguous:
            # Was the plain model (without connectivity) feasible at all?
            plain, _x, _obj = _build_problem(
                order, conflict_edges, stitches, adjacency, False
            )
            plain.solve(_cbc())
            if _status(plain) == "Optimal":
                result["reason"] = "disconnected"
        return result
    if status != "Optimal":
        raise SolverError(f"求解器未能求得最优解（状态：{status}）")

    # Read the optimum off the optimal coloring itself: stitch weights are
    # positive integers, so the cut total is an exact integer.
    first_colors = [_color_of(x, i) for i in range(n)]
    best = sum(w for a, b, w in stitches if first_colors[pos[a]] != first_colors[pos[b]])
    prob += objective <= best + 1e-6, "optimal_bound"

    # Lexicographically smallest canonical optimum: minimize the mask at
    # each position in turn (ids ascending), then pin it before moving on.
    colors = [0] * n
    for i in range(n):
        prob.setObjective(pulp.lpSum(k * x[i][k] for k in MASKS))
        prob.solve(_cbc())
        if _status(prob) != "Optimal":
            raise SolverError("求解器在构造字典序最小方案时失败")
        colors[i] = _color_of(x, i)
        prob += x[i][colors[i]] == 1, f"fix_{i}"

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
            witness["mask_islands"] = _adopted_adjacency(
                adjacency, pos, witness_colors
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
        result["contiguous"] = True
        result["mask_islands"] = _adopted_adjacency(adjacency, pos, colors)
    return result

"""API-level tests: validation, itemized errors, and result shapes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_solve_optimal():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[1, 2], [2, 3], [1, 3]],
            "stitch_edges": [{"pair": [3, 4], "weight": 5}, [1, 4, 1]],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "optimal"
    assert body["objective"] == 1
    assert body["unique"] is True
    assert body["assignment"] == {"1": 0, "2": 1, "3": 2, "4": 2}
    assert body["cut_stitches"] == [{"pair": [1, 4], "weight": 1}]


def test_solve_infeasible_is_distinguished():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[a, b] for a in range(1, 5) for b in range(a + 1, 5)],
            "stitch_edges": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "infeasible"


def test_invalid_input_returns_itemized_errors():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 1, 2, "x"],
            "conflict_edges": [[1, 9], [2, 2], [1, 2], [1, 2]],
            "stitch_edges": [{"pair": [1, 2], "weight": 3}, {"pair": [2, 3], "weight": 0}],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "invalid"
    errors = body["errors"]
    assert len(errors) >= 8
    assert all(set(e) == {"loc", "message"} for e in errors)
    locs = {e["loc"] for e in errors}
    # duplicate id, non-integer id, fragment count
    assert "fragments[1]" in locs
    assert "fragments[3]" in locs
    assert "fragments" in locs
    # unknown endpoint, self-loop, duplicate pair
    assert "conflict_edges[0]" in locs
    assert "conflict_edges[1]" in locs
    assert "conflict_edges[3]" in locs
    # pair in both edge types, non-positive weight
    assert "stitch_edges[0]" in locs
    assert "stitch_edges[1]" in locs


def test_fragment_count_bounds():
    too_few = client.post("/api/solve", json={"fragments": [1, 2, 3]})
    assert too_few.status_code == 400
    too_many = client.post("/api/solve", json={"fragments": list(range(1, 50))})
    assert too_many.status_code == 400
    assert any("4–48" in e["message"] for e in too_many.json()["errors"])


def test_non_object_body_is_invalid():
    resp = client.post("/api/solve", json=[1, 2, 3])
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid"


def test_witness_returned_when_multiple_optima():
    resp = client.post("/api/solve", json={"fragments": [1, 2, 3, 4]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "optimal"
    assert body["unique"] is False
    assert body["witness"]["assignment"] != body["assignment"]


# --------------------------------------------------------------------------
# Continuous mask islands
# --------------------------------------------------------------------------

CONTIGUOUS_PAYLOAD = {
    "fragments": [1, 2, 3, 4],
    "conflict_edges": [[1, 2]],
    "stitch_edges": [[3, 4, 1]],
    "contiguous": True,
    "adjacency_edges": [[1, 3], [2, 4]],
}


def test_contiguous_optimal_response():
    resp = client.post("/api/solve", json=CONTIGUOUS_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "optimal"
    assert body["objective"] == 1
    assignment = body["assignment"]
    assert assignment["1"] == assignment["3"]
    assert assignment["2"] == assignment["4"]
    summary = body["connectivity"]
    assert summary["enabled"] is True
    covered = [f for m in summary["masks"] for f in m["fragments"]]
    assert sorted(covered) == [1, 2, 3, 4]
    used = [tuple(e) for m in summary["masks"] for e in m["adjacency_edges"]]
    assert set(used) <= {(1, 3), (2, 4)}
    assert body["witness"] is None or "connectivity" in body["witness"]


def test_contiguous_connectivity_blocked():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[1, 2], [2, 3], [1, 3]],
            "contiguous": True,
            "adjacency_edges": [[1, 2]],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["reason"] == "connectivity_blocked"


def test_contiguous_conflict_graph_infeasible():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[a, b] for a in range(1, 5) for b in range(a + 1, 5)],
            "contiguous": True,
            "adjacency_edges": [[1, 2], [2, 3], [3, 4]],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["reason"] == "conflict_graph"


def test_contiguous_off_stays_byte_compatible():
    # Explicit false and missing flag alike keep the original shape.
    for payload in (
        {"fragments": [1, 2, 3, 4], "contiguous": False},
        {"fragments": [1, 2, 3, 4]},
    ):
        resp = client.post("/api/solve", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "connectivity" not in body
        assert "contiguous" not in body

    infeasible = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[a, b] for a in range(1, 5) for b in range(a + 1, 5)],
        },
    )
    assert infeasible.json() == {"status": "infeasible"}


def test_adjacency_validation_is_itemized():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "contiguous": True,
            "adjacency_edges": [
                [1, 9],   # unknown fragment
                [2, 2],   # self loop
                [1, 2],
                [2, 1],   # duplicate undirected pair
                [3, 4],
            ],
        },
    )
    assert resp.status_code == 400
    errors = resp.json()["errors"]
    locs = {e["loc"] for e in errors}
    assert "adjacency_edges[0]" in locs
    assert "adjacency_edges[1]" in locs
    assert "adjacency_edges[3]" in locs


def test_contiguous_requires_at_least_one_adjacency_edge():
    resp = client.post(
        "/api/solve",
        json={"fragments": [1, 2, 3, 4], "contiguous": True, "adjacency_edges": []},
    )
    assert resp.status_code == 400
    locs = {e["loc"] for e in resp.json()["errors"]}
    assert "adjacency_edges" in locs


def test_non_boolean_contiguous_is_rejected():
    resp = client.post(
        "/api/solve",
        json={"fragments": [1, 2, 3, 4], "contiguous": "yes"},
    )
    assert resp.status_code == 400
    locs = {e["loc"] for e in resp.json()["errors"]}
    assert "contiguous" in locs


def test_adjacency_ignored_when_mode_off():
    # Junk adjacency data must not affect an off-mode request at all.
    resp = client.post(
        "/api/solve",
        json={"fragments": [1, 2, 3, 4], "adjacency_edges": [[1, 9], [2, 2]]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "optimal"

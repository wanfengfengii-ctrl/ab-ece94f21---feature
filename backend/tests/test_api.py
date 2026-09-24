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


def test_legacy_request_omits_contiguous_fields():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[1, 2], [2, 3], [1, 3]],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "contiguous" not in body
    assert "mask_islands" not in body


def test_contiguous_feasible_returns_islands():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[1, 2], [2, 3], [1, 3]],
            "adjacency_edges": [[1, 2], [3, 4]],
            "contiguous": True,
        },
    )
    assert resp.status_code == 200, f"{resp.text}"
    body = resp.json()
    assert body["status"] == "optimal"
    assert body["contiguous"] is True
    by_mask = {island["mask"]: island for island in body["mask_islands"]}
    assert set(by_mask) == {0, 1, 2}
    assert by_mask[2]["fragments"] == [3, 4]
    assert by_mask[2]["adjacency_edges"] == [{"pair": [3, 4]}]
    # Every fragment is covered exactly once across the islands.
    covered = [f for island in body["mask_islands"] for f in island["fragments"]]
    assert sorted(covered) == [1, 2, 3, 4]


def test_contiguous_blockage_reports_disconnected():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "conflict_edges": [[1, 2], [2, 3], [1, 3]],
            "adjacency_edges": [[1, 2]],
            "contiguous": True,
        },
    )
    assert resp.status_code == 200, f"{resp.text}"
    assert resp.json() == {"status": "infeasible", "reason": "disconnected"}


def test_mode_off_ignores_adjacency_edges():
    # With the mode off, adjacency data must not constrain anything: the
    # response is identical to the legacy request without adjacency data.
    payload = {
        "fragments": [1, 2, 3, 4],
        "conflict_edges": [[1, 2], [2, 3], [1, 3]],
    }
    legacy = client.post("/api/solve", json=payload).json()
    with_edges = client.post(
        "/api/solve", json={**payload, "adjacency_edges": [[1, 2]], "contiguous": False}
    ).json()
    assert with_edges == legacy
    assert "mask_islands" not in with_edges


def test_adjacency_validation_errors_are_itemized():
    resp = client.post(
        "/api/solve",
        json={
            "fragments": [1, 2, 3, 4],
            "adjacency_edges": [[1, 9], [2, 2], [1, 2], [1, 2]],
            "contiguous": True,
        },
    )
    assert resp.status_code == 400
    locs = {e["loc"] for e in resp.json()["errors"]}
    assert "adjacency_edges[0]" in locs  # unknown fragment
    assert "adjacency_edges[1]" in locs  # self loop
    assert "adjacency_edges[3]" in locs  # duplicate undirected pair
    # The valid pair must not be flagged.
    assert "adjacency_edges[2]" not in locs


def test_adjacency_edges_still_validated_when_mode_is_off():
    # Data hygiene applies regardless of the switch; stale edge data must be
    # rejected even while it would not constrain the solve.
    resp = client.post(
        "/api/solve",
        json={"fragments": [1, 2, 3, 4], "adjacency_edges": [[1, 9]]},
    )
    assert resp.status_code == 400
    assert any(
        e["loc"] == "adjacency_edges[0]" for e in resp.json()["errors"]
    )


def test_non_boolean_contiguous_is_invalid():
    resp = client.post(
        "/api/solve",
        json={"fragments": [1, 2, 3, 4], "contiguous": "yes"},
    )
    assert resp.status_code == 400
    locs = {e["loc"] for e in resp.json()["errors"]}
    assert "contiguous" in locs

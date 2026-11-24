def test_sabetagal_schemas_importable():
    from cellquant.api.schemas.sabetagal import SABetaGalRunRequest
    req = SABetaGalRunRequest(session_id="abc")
    assert req.session_id == "abc"
    assert req.white_percentile == 95.0
    assert req.threshold is None  # Otsu by default


def test_sabetagal_router_registered():
    from cellquant.api.app import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/v1/sabetagal/run" in paths
    assert "/api/v1/sabetagal/summary/{session_id}" in paths


def test_worker_task_importable():
    from cellquant.tasks.worker import run_sabetagal_task
    assert callable(run_sabetagal_task)

"""API surface of the trust tier: catalog, findings and consensus."""
from fastapi.testclient import TestClient

from verity.models.findings import Classification, Finding, Reasoning
from verity_api.main import app
from verity_api.service import consensus


def _finding(det: str, cls: Classification, tier: str, conf: float = 0.9) -> Finding:
    return Finding(detector_id=det, detector_version="1", classification=cls, confidence=conf,
                   reasoning=Reasoning(summary="s"), tier=tier,
                   raw_output={"prediction": 0.9 if cls is Classification.FAKE else 0.1})


class TestConsensus:
    def test_untrained_never_votes(self):
        c = consensus([
            _finding("a", Classification.REAL, "trained"),
            _finding("b", Classification.REAL, "trained"),
            _finding("c", Classification.FAKE, "random-init"),
            _finding("d", Classification.FAKE, "backbone-only"),
        ])
        assert c["total"] == 2 and c["fake"] == 0 and c["real"] == 2
        assert c["untrained_excluded"] == 2
        assert c["band"] == "strong_authentic"

    def test_only_untrained_is_no_analysis(self):
        c = consensus([_finding("c", Classification.FAKE, "random-init")])
        assert c["band"] == "no_analysis" and c["untrained_excluded"] == 1


class TestCatalog:
    def test_detectors_expose_tier(self):
        client = TestClient(app)
        body = client.get("/api/detectors").json()
        assert body["count"] >= 1
        for entry in body["detectors"]:
            assert entry["tier"] in {"trained", "backbone-only", "random-init", "untrained"}
            assert entry["trained"] == (entry["tier"] == "trained")

    def test_run_request_accepts_include_untrained(self):
        from verity_api.schemas import RunRequest
        assert RunRequest(media_id="x").include_untrained is False
        assert RunRequest(media_id="x", include_untrained=True).include_untrained is True


class TestJobGuard:
    def test_second_job_for_same_case_is_refused(self):
        import threading
        import pytest
        from verity_api.jobs import JobConflict, JobManager

        manager = JobManager(max_workers=1)
        gate = threading.Event()
        first = manager.submit("run", "first", lambda ctx: gate.wait(5), media_ids=["abc"])
        try:
            with pytest.raises(JobConflict):
                manager.submit("report", "second", lambda ctx: None, media_ids=["abc"])
            # another case is fine (it queues behind the single worker)
            other = manager.submit("report", "other", lambda ctx: None, media_ids=["xyz"])
            assert other.id != first.id
        finally:
            gate.set()

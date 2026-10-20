import pytest
from pathlib import Path
from verity.models.media import MediaType, MediaInfo, MediaCopy
from verity.models.artifacts import ArtifactType, Artifact, ArtifactCollection
from verity.models.findings import Classification, Reasoning, Finding, FindingsCollection
from verity.models.report import Report


class TestMediaModels:
    def test_media_info_defaults(self):
        info = MediaInfo(original_path=Path("/test/video.mp4"))
        assert info.media_type == MediaType.UNKNOWN
        assert info.checksum == ""

    def test_media_copy_created(self):
        info = MediaInfo(original_path=Path("/test/v.mp4"), media_type=MediaType.VIDEO)
        copy = MediaCopy(media_id="abc123", original=info, copy_path=Path("/workspace/media/abc123.mp4"))
        assert copy.verified is False
        assert copy.media_id == "abc123"


class TestArtifactModels:
    def test_artifact_creation(self):
        art = Artifact(
            artifact_type=ArtifactType.FACE_CROP,
            path=Path("/workspace/artifacts/face.png"),
            source_media_id="abc123",
        )
        assert art.artifact_type == ArtifactType.FACE_CROP

    def test_artifact_collection(self):
        col = ArtifactCollection(media_id="abc123")
        assert len(col.artifacts) == 0


class TestFindingModels:
    def test_finding_default_classification(self):
        reasoning = Reasoning(summary="Test reasoning")
        finding = Finding(
            detector_id="xception",
            detector_version="1.0.0",
            classification=Classification.FAKE,
            confidence=0.95,
            reasoning=reasoning,
        )
        assert finding.classification == Classification.FAKE
        assert finding.confidence == 0.95
        assert finding.reasoning.summary == "Test reasoning"

    def test_findings_collection_aggregation(self):
        col = FindingsCollection(media_id="abc123")
        assert col.aggregated_score is None


class TestReportModel:
    def test_report_creation(self):
        info = MediaInfo(original_path=Path("/test/v.mp4"), media_type=MediaType.VIDEO)
        arts = ArtifactCollection(media_id="abc123")
        finds = FindingsCollection(media_id="abc123")
        report = Report(media=info, artifacts=arts, findings=finds)
        assert report.summary == ""

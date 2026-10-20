"""HTTP API for Verity.

Wraps the :mod:`verity` pipeline — the same objects the ``verity`` CLI drives —
behind a job-oriented REST API so a browser UI can expose every CLI capability
without shelling out. The CLI remains the reference implementation; this
package adds no detection logic of its own.
"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # pragma: no cover - thin re-export
    from verity_api.main import create_app as _create_app

    return _create_app(*args, **kwargs)

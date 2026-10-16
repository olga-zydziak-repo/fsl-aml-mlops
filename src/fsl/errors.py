"""Readable component errors — layer B.

The problem this solves: when a component fails inside a Vertex Custom Job, you
see a raw Python traceback in Cloud Logging, often with the real cause buried
(a GCS PermissionDenied, an OOM, an HPO run with zero successful trials). You
then reverse-engineer "what do I actually check?" from a stack trace — which is
exactly what we kept doing from screenshots during the HPO saga.

`explain_failure` wraps an operation and, when it raises, re-raises with a
message that names the likely cause and the concrete thing to check — WITHOUT
swallowing the original (it's chained via `raise ... from`, so the full
traceback is still there for anyone who wants it).

Usage inside a component:

    from fsl.errors import explain_failure

    with explain_failure("saving model to GCS", bucket=bucket):
        gcs.blob(path).upload_from_filename(local)

If the upload raises PermissionDenied, the surfaced message becomes:

    saving model to GCS failed: permission denied on bucket 'X'.
    The job's service account needs storage.objectAdmin (or objectCreator) on
    that bucket. Original error: 403 ...

The matching is heuristic (substring on the exception text) and deliberately
conservative: if nothing matches, it re-raises with the operation label
prepended and nothing invented. Better a truthful "operation X failed: <error>"
than a confident wrong diagnosis.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Optional


# (substring to look for in the error text, template for the guidance).
# {op}, {bucket}, {detail} are filled where available. Order matters: first
# match wins, so put more-specific patterns before generic ones.
_HINTS: list[tuple[str, str]] = [
    ("permissiondenied",
     "{op} failed: permission denied. The job's service account likely lacks "
     "the needed IAM role. For GCS: storage.objectAdmin on '{bucket}'. For "
     "Model Registry: aiplatform.user + model upload/update. For creating "
     "jobs (HPO): aiplatform.user. Original error: {detail}"),
    ("403",
     "{op} failed: access denied (403). Check the service account's IAM roles "
     "for this resource. Original error: {detail}"),
    ("404",
     "{op} failed: resource not found (404). Check the path/name and that it "
     "was created by an earlier step. Original error: {detail}"),
    ("not found",
     "{op} failed: something wasn't found. If this is GCS, check the object "
     "path and that the producing step ran. Original error: {detail}"),
    ("out of memory",
     "{op} failed: out of memory. Reduce batch/episode size, or give the "
     "Custom Job a larger machine_type. Original error: {detail}"),
    ("cuda out of memory",
     "{op} failed: GPU out of memory. Lower episode/query size or use a larger "
     "GPU. Original error: {detail}"),
    ("no space left",
     "{op} failed: disk full in the container. Increase boot_disk_size_gb on "
     "the Custom Job. Original error: {detail}"),
    ("quota",
     "{op} failed: a quota was exceeded. For GPUs this is usually the "
     "'Custom model training' accelerator quota in Vertex Training (separate "
     "from Compute Engine). Request more, or switch to CPU. Original error: {detail}"),
    ("deadlineexceeded",
     "{op} failed: timed out. A dependency may be slow or hung; retrying once "
     "is reasonable if it's a transient network/GCS blip. Original error: {detail}"),
    ("connection",
     "{op} failed: a network/connection error. Often transient — a single "
     "retry is reasonable. Original error: {detail}"),
]


class ComponentError(RuntimeError):
    """A component failure re-raised with actionable guidance. Chains the original."""


@contextlib.contextmanager
def explain_failure(
    operation: str,
    *,
    bucket: Optional[str] = None,
) -> Iterator[None]:
    """Wrap an operation; on failure re-raise with a cause-specific hint.

    The original exception is preserved as the __cause__ (via `raise ... from`),
    so no information is lost — this only ADDS a readable, actionable summary on
    top. If no hint pattern matches, the operation label is prepended and the
    original text is passed through verbatim; nothing is fabricated.

    Args:
        operation: human phrase for what was being attempted, e.g.
            "saving model to GCS" or "registering model". Used in the message.
        bucket: optional bucket name, filled into GCS-related hints.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - intentional: annotate then re-raise
        text = str(exc).lower()
        detail = f"{type(exc).__name__}: {exc}"
        for needle, template in _HINTS:
            if needle in text or needle in type(exc).__name__.lower():
                msg = template.format(op=operation, bucket=bucket or "<bucket>",
                                      detail=detail)
                raise ComponentError(msg) from exc
        # no specific match: truthful fallback, original preserved
        raise ComponentError(f"{operation} failed: {detail}") from exc


def require(condition: bool, message: str) -> None:
    """Assert a precondition inside a component with a clear message.

    Use for input validation at the start of a component — e.g. a model dir
    that must exist, an accuracy that must be in range — so the failure is a
    named precondition, not a downstream AttributeError.

    Raises:
        ComponentError: if condition is falsy.
    """
    if not condition:
        raise ComponentError(f"precondition failed: {message}")

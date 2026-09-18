"""Shared fixtures.

`results/*.pt` is gitignored, so a fresh clone has no trained policy. Tests
that exercise the real brain end to end need one. Skipping them with the
loader's own rejection reason keeps the suite honest: a missing artifact is
reported as a missing artifact, not as a passing test and not as a
mysterious failure.
"""
import pytest


@pytest.fixture(scope="session")
def fly_service():
    from serving.serve import FlyService

    svc = FlyService()
    if not svc.policies:
        reasons = "; ".join(
            f"{name}: {info.get('reason')}"
            for name, info in sorted(svc.rejected.items())) or "none found"
        pytest.skip(
            "no servable trained policy checkpoint in results/ "
            f"({reasons}). Run `python -m learning.train` to produce one.")
    return svc

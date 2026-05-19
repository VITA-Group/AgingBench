"""Tests for the S8 SWE-bench-Aging Docker runner (Phase 2).

Most tests are credential-free unit tests against the helper functions
+ the lifecycle scheduler. The full end-to-end smoke (real `docker run`
on a sphinx image) is gated behind `S8_LIVE_DOCKER_SMOKE=1` so CI
doesn't pull/run 2.6 GB of Docker per test invocation.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from agingbench.generators.pressure_config import PressureConfig
from agingbench.scenarios.s8_swe_bench.docker_runner import (
    docker_available,
    image_exists_locally,
    resolve_image_for_instance,
)
from agingbench.scenarios.s8_swe_bench.lifecycle import (
    DEP_BUMP_CANDIDATES,
    LifecycleScheduler,
)


# ---- pure helpers ---------------------------------------------------------

def test_resolve_image_for_instance_substitutes_pattern():
    assert resolve_image_for_instance(
        "sphinx-doc__sphinx-7454",
        "sweb.eval.x86_64.{instance_id}:latest",
    ) == "sweb.eval.x86_64.sphinx-doc__sphinx-7454:latest"


def test_resolve_image_for_instance_handles_other_pattern():
    assert resolve_image_for_instance(
        "django__django-12345",
        "myreg/{instance_id}:v1",
    ) == "myreg/django__django-12345:v1"


# ---- lifecycle scheduler --------------------------------------------------

def test_lifecycle_scheduler_no_pressure_no_events():
    """forget_rate=0, update_rate=0 -> no scheduled events."""
    p = PressureConfig.none()  # forget_rate=0, update_rate=0
    sched = LifecycleScheduler(pressure=p, n_sessions=10, seed=42)
    events = sched.schedule()
    assert events == []


def test_lifecycle_scheduler_heavy_more_events_than_light():
    """Heavier pressure should produce more lifecycle events on average."""
    light = LifecycleScheduler(
        pressure=PressureConfig.light(), n_sessions=20, seed=42,
    ).schedule()
    heavy = LifecycleScheduler(
        pressure=PressureConfig.heavy(), n_sessions=20, seed=42,
    ).schedule()
    assert len(heavy) > len(light)


def test_lifecycle_scheduler_warmup_blocks_early_events():
    """warmup_sessions=5 -> no events before session 5."""
    p = PressureConfig(
        tokens_per_session=1000, dependency_density=0.0, update_rate=1.0,
        max_chain_depth=1, n_confusable_pairs=0, confusable_start_session=0,
        warmup_sessions=5, forget_rate=1.0,
    )
    events = LifecycleScheduler(pressure=p, n_sessions=10, seed=42).schedule()
    assert all(e.session >= 5 for e in events), (
        f"events fired before warmup: {[e for e in events if e.session < 5]}"
    )


def test_lifecycle_scheduler_seeds_are_deterministic():
    """Same seed -> same scheduled events."""
    p = PressureConfig.medium()
    a = LifecycleScheduler(pressure=p, n_sessions=10, seed=42).schedule()
    b = LifecycleScheduler(pressure=p, n_sessions=10, seed=42).schedule()
    assert [(e.session, e.event_type) for e in a] == \
           [(e.session, e.event_type) for e in b]


def test_dep_bump_candidate_list_is_realistic():
    """Sanity: every candidate is a real Python package the container has."""
    assert "pytest" in DEP_BUMP_CANDIDATES
    for pkg in DEP_BUMP_CANDIDATES:
        assert pkg.replace("_", "").replace("-", "").isalnum(), pkg


# ---- runner: precondition check (no real containers) ---------------------

def test_runner_precondition_check_reports_missing_images(tmp_path):
    """Runner should report which images aren't on the host."""
    from agingbench.runner.s8_runner import S8SweBenchRunner, S8RunnerConfig
    from pathlib import Path

    cfg = S8RunnerConfig(
        seed=42,
        n_sessions=2,
        pressure=PressureConfig.medium(),
        sut_id="fixture",
        docker_image_pattern="definitely.not.a.real.image.{instance_id}:nope",
        workspace_root=tmp_path / "ws",
    )
    runner = S8SweBenchRunner(cfg)
    report = runner.precondition_check()
    assert isinstance(report["docker_available"], bool)
    assert "missing_images" in report
    assert report["all_images_present"] is False  # bogus pattern guarantees miss


# ---- live smoke (gated) ---------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("S8_LIVE_DOCKER_SMOKE"),
    reason="requires S8_LIVE_DOCKER_SMOKE=1 + 8 sphinx images cached",
)
def test_s8_live_one_session_smoke(tmp_path):
    """End-to-end smoke: spin one container, run an exec, write+read /agentmemory.

    Gated to avoid 2.6 GB Docker pull on every CI run. To enable:
        S8_LIVE_DOCKER_SMOKE=1 pytest tests/test_s8_docker_runner.py
    """
    from agingbench.scenarios.s8_swe_bench.docker_runner import S8DockerSession

    image = "sweb.eval.x86_64.sphinx-doc__sphinx-7454:latest"
    if not image_exists_locally(image):
        pytest.skip(f"image not cached: {image}")
    if not docker_available():
        pytest.skip("docker daemon not reachable")

    memdir = tmp_path / "memory"
    with S8DockerSession(image=image, memory_dir=memdir,
                         instance_id="sphinx-doc__sphinx-7454") as session:
        # Container is alive.
        assert session.container_id

        # Repo present at /testbed (sphinx checked out at base_commit).
        r = session.exec("ls /testbed | head -3")
        assert r.exit_code == 0
        assert r.stdout.strip()

        # /agentmemory is writable from BOTH directions (container start
        # chmod'd it 0777 specifically to enable this).
        session.write_memory_file(".aging/host_write.txt", "from host")
        # Container can read what host wrote.
        r1 = session.exec("cat /agentmemory/.aging/host_write.txt")
        assert r1.exit_code == 0 and r1.stdout.strip() == "from host"
        # Container can also write; host can read.
        session.exec("echo from_container > /agentmemory/.aging/container_write.txt")
        assert (memdir / ".aging" / "container_write.txt").read_text().strip() == "from_container"

    # After context exit, container is gone.
    import subprocess as _sp
    proc = _sp.run(["docker", "ps", "-a", "-q", "-f", f"name={session.container_name}"],
                   capture_output=True, text=True)
    assert proc.stdout.strip() == "", f"container leaked: {proc.stdout!r}"

    # Memory dir survives on host (this is the persistence guarantee).
    assert (memdir / ".aging" / "host_write.txt").is_file()
    assert (memdir / ".aging" / "container_write.txt").is_file()

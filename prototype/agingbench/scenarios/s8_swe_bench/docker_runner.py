"""S8 SWE-bench-Aging — per-session Docker container lifecycle.

Wraps the `docker` CLI (no Python SDK dep) for one session at a time.
Each session:
  1. Resolve the per-issue image (from the chain spec)
  2. Spin a container with the agent's persistent /agentmemory volume
     mounted from the host (this is the agent's self-planned memory
     — notes.md, scratch files, anything the agent chooses to write
     under /agentmemory/.aging/)
  3. Hand the container_id off to the agent layer (Phase 3)
  4. Optionally run the upstream `run-tests.sh` for verification (Phase 4)
  5. Tear down (always, even on agent failure)

The repo at /testbed is FRESH per session (each container is built from
its issue's pinned base_commit) — that's the SWE-bench contract. The
ONLY thing that persists across sessions is /agentmemory.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_EXEC_TIMEOUT_SEC = 300
DEFAULT_RUN_TIMEOUT_SEC = 1800
AGENT_MEMORY_MOUNT = "/agentmemory"


@dataclass
class ExecResult:
    """One `docker exec` invocation result."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float


@dataclass
class S8DockerSession:
    """One session's container handle.

    Use as a context manager so cleanup is guaranteed:

        with S8DockerSession(image, memory_dir) as session:
            session.exec("ls /testbed")
            ...
    """
    image: str
    memory_dir: Path                  # host path; mounted at /agentmemory
    instance_id: str                  # for naming + diagnostics
    container_name: str = field(default="", init=False)
    container_id: str = field(default="", init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.memory_dir = Path(self.memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # Unique container name per session — prevents collisions on parallel runs.
        self.container_name = f"agingbench_s8_{self.instance_id}_{uuid.uuid4().hex[:8]}"

    # ---- context manager ------------------------------------------------

    def __enter__(self) -> "S8DockerSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tear_down()

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> str:
        """Spin the container detached; mount /agentmemory."""
        if self._started:
            raise RuntimeError(f"Session {self.container_name} already started")
        cmd = [
            "docker", "run",
            "-d",                                # detached
            "--rm",                              # auto-remove on exit
            "--name", self.container_name,
            "-v", f"{self.memory_dir.absolute()}:{AGENT_MEMORY_MOUNT}",
            "--workdir", "/testbed",
            self.image,
            "sleep", "infinity",                 # keep it alive for exec
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {self.image}:\n"
                f"  stderr: {result.stderr}"
            )
        self.container_id = result.stdout.strip()
        self._started = True
        # Make /agentmemory world-writable so the host (running as the
        # invoking user) can manipulate / inspect / clean up files the
        # container later creates as root. Without this, host-side
        # workspace_flush, write_memory_file, and shutil.rmtree all hit
        # PermissionError on container-created subdirectories.
        self._exec_raw(
            ["docker", "exec", self.container_name,
             "bash", "-c", f"mkdir -p {AGENT_MEMORY_MOUNT} && chmod -R 0777 {AGENT_MEMORY_MOUNT}"],
            timeout=15,
        )
        return self.container_id

    def tear_down(self) -> None:
        """Best-effort cleanup; safe to call repeatedly."""
        if not self._started:
            return
        # `docker stop` triggers --rm cleanup since we ran with that flag.
        subprocess.run(
            ["docker", "stop", "--time", "5", self.container_name],
            capture_output=True, text=True, timeout=30,
        )
        # If --rm didn't fire (e.g., container already gone), force-remove.
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True, text=True, timeout=10,
        )
        self._started = False

    # ---- exec interface -------------------------------------------------

    @staticmethod
    def _exec_raw(argv: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
        """Helper for internal docker invocations that don't return ExecResult."""
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    def cp_to_container(self,
                         host_path: Path,
                         container_path: str,
                         timeout_sec: int = 120) -> ExecResult:
        """Copy a host path INTO the container. Inverse of cp_to_host.

        Used by S9 to inject synthetic consistency tests into /testbed
        before pytest runs.
        """
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        host_path = Path(host_path)
        argv = ["docker", "cp", str(host_path),
                f"{self.container_name}:{container_path}"]
        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec)
        dt = time.time() - t0
        return ExecResult(
            command=shlex.join(argv),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_sec=round(dt, 3),
        )

    def cp_to_host(self,
                   container_path: str,
                   host_path: Path,
                   timeout_sec: int = 120) -> ExecResult:
        """Copy a path FROM the container TO the host.

        Used to give the agent read access to /testbed without giving
        write access (the agent operates on the copy; we apply the
        agent's diff back inside the container).
        """
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        host_path = Path(host_path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        argv = ["docker", "cp",
                f"{self.container_name}:{container_path}",
                str(host_path)]
        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec)
        dt = time.time() - t0
        return ExecResult(
            command=shlex.join(argv),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_sec=round(dt, 3),
        )

    def exec(self,
             command: str | list[str],
             timeout_sec: int = DEFAULT_EXEC_TIMEOUT_SEC,
             user: Optional[str] = None,
             cwd: Optional[str] = None) -> ExecResult:
        """Run a command in the container; return stdout/stderr/exit.

        `command` can be a single string (run via `bash -c`) or a list
        of argv tokens (run directly).
        """
        if not self._started:
            raise RuntimeError("session not started; call start() first")
        argv = ["docker", "exec"]
        if user:
            argv += ["-u", user]
        if cwd:
            argv += ["-w", cwd]
        argv.append(self.container_name)
        if isinstance(command, str):
            argv += ["bash", "-c", command]
        else:
            argv += list(command)

        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec)
        dt = time.time() - t0
        return ExecResult(
            command=command if isinstance(command, str) else shlex.join(command),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_sec=round(dt, 3),
        )

    # ---- memory helpers (convenience over exec) ------------------------

    def read_memory_file(self, relative_path: str) -> Optional[str]:
        """Read /agentmemory/<relative_path> from the host side (faster than exec)."""
        p = self.memory_dir / relative_path
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8", errors="replace")

    def write_memory_file(self, relative_path: str, content: str) -> Path:
        """Write to /agentmemory/<relative_path> from the host (visible inside container)."""
        p = self.memory_dir / relative_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def memory_size_bytes(self) -> int:
        total = 0
        for f in self.memory_dir.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total


# ---- helpers --------------------------------------------------------------

def image_exists_locally(image: str) -> bool:
    """True iff `docker image inspect <image>` succeeds."""
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, timeout=15,
    )
    return proc.returncode == 0


def resolve_image_for_instance(instance_id: str, pattern: str) -> str:
    """Apply the chain's `docker_image_pattern` to an instance_id."""
    return pattern.format(instance_id=instance_id)


def docker_available() -> bool:
    """Quick probe: does `docker info` work without sudo?"""
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

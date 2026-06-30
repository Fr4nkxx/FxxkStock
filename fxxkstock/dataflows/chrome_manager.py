"""Cross-platform lifecycle management for a local Google Chrome CDP process."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ("macos", "windows", "ubuntu")
_START_LOCK = threading.Lock()
_STARTED_PROCESS: subprocess.Popen | None = None
_STARTED_PLATFORM: str | None = None
_ACTIVE_LEASES = 0
_DIRECT_OPENER = build_opener(ProxyHandler({}))


def current_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if os.name == "nt":
        return "windows"
    return "ubuntu"


class ChromeManager:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.platform = str(cfg.get("cn_browser_platform") or current_platform()).lower()
        self.executable = cfg.get("cn_browser_executable")
        self.profile_dir = Path(
            cfg.get("cn_browser_profile_dir", "browser_data/chrome-profile")
        ).expanduser()
        self.cdp_url = str(cfg.get("cn_browser_cdp_url", "http://127.0.0.1:9222")).rstrip("/")
        self.timeout = float(cfg.get("cn_browser_startup_timeout_seconds", 15))

    def _cdp_endpoint(self) -> str:
        parsed = urlparse(self.cdp_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Chrome CDP must listen on localhost")
        return f"{self.cdp_url}/json/version"

    def is_cdp_available(self, timeout: float = 0.75) -> bool:
        try:
            # CDP is always local. Never let HTTP(S)_PROXY or ALL_PROXY route
            # this health check through the user's outbound proxy.
            with _DIRECT_OPENER.open(self._cdp_endpoint(), timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("webSocketDebuggerUrl"))
        except Exception:
            return False

    def resolve_executable(self) -> Path | None:
        if self.platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"unsupported Chrome platform: {self.platform}")
        if self.executable:
            path = Path(str(self.executable)).expanduser()
            return path if path.is_file() else None

        if self.platform == "macos":
            candidates = [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ]
            return next((path for path in candidates if path.is_file()), None)

        if self.platform == "windows":
            roots = [
                os.environ.get("PROGRAMFILES"),
                os.environ.get("PROGRAMFILES(X86)"),
                os.environ.get("LOCALAPPDATA"),
            ]
            candidates = [
                Path(root) / "Google/Chrome/Application/chrome.exe"
                for root in roots if root
            ]
            return next((path for path in candidates if path.is_file()), None)

        for command in ("google-chrome", "google-chrome-stable"):
            resolved = shutil.which(command)
            if resolved:
                return Path(resolved)
        candidates = [
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/google-chrome-stable"),
        ]
        return next((path for path in candidates if path.is_file()), None)

    def build_command(self, executable: Path) -> list[str]:
        parsed = urlparse(self.cdp_url)
        port = parsed.port or 9222
        return [
            str(executable),
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self.profile_dir.resolve()}",
            "--no-first-run",
            "--no-default-browser-check",
        ]

    def status(self) -> dict[str, Any]:
        global _STARTED_PROCESS, _STARTED_PLATFORM, _ACTIVE_LEASES
        running = self.is_cdp_available()
        managed = bool(
            running
            and _STARTED_PROCESS is not None
            and _STARTED_PROCESS.poll() is None
        )
        return {
            "available": running,
            "platform": self.platform,
            "managed": managed,
            "managed_platform": _STARTED_PLATFORM if managed else None,
            "cdp_url": self.cdp_url,
            "active_leases": _ACTIVE_LEASES if managed else 0,
            "profile_dir": str(self.profile_dir.resolve()),
            "profile_exists": self.profile_dir.exists(),
            "can_close": bool(managed and _ACTIVE_LEASES <= 1),
        }

    def open_url(self, url: str) -> dict[str, Any]:
        """Open a new tab in the running CDP Chrome."""
        if not self.is_cdp_available():
            return {**self.status(), "state": "not_running"}
        endpoint = f"{self.cdp_url}/json/new?{quote(url, safe='')}"
        request = Request(endpoint, method="PUT")
        try:
            with _DIRECT_OPENER.open(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {
                **self.status(),
                "state": "opened",
                "target_id": payload.get("id"),
            }
        except Exception as exc:
            return {
                **self.status(),
                "state": "open_failed",
                "message": f"failed to open login site: {exc}",
            }

    def _request_graceful_close(self) -> bool:
        """Ask Chrome to flush its profile and exit through CDP."""
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.connect_over_cdp(
                    self.cdp_url,
                    timeout=max(1000, int(self.timeout * 1000)),
                )
                browser.close()
            return True
        except Exception as exc:
            logger.debug("Graceful Chrome close failed, using process fallback: %s", exc)
            return False

    def ensure_running(self) -> dict[str, Any]:
        global _STARTED_PROCESS, _STARTED_PLATFORM, _ACTIVE_LEASES
        with _START_LOCK:
            if self.is_cdp_available():
                if _STARTED_PROCESS is not None and _STARTED_PROCESS.poll() is None:
                    _ACTIVE_LEASES += 1
                return {**self.status(), "state": "already_running"}
            if self.platform != current_platform():
                return {
                    **self.status(),
                    "state": "failed_fallback",
                    "message": (
                        f"selected Chrome platform {self.platform} does not match "
                        f"runtime platform {current_platform()}"
                    ),
                }
            try:
                executable = self.resolve_executable()
            except ValueError as exc:
                return {**self.status(), "state": "failed_fallback", "message": str(exc)}
            if executable is None:
                return {
                    **self.status(),
                    "state": "failed_fallback",
                    "message": f"Google Chrome executable not found for {self.platform}",
                }

            self.profile_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
            }
            child_env = os.environ.copy()
            local_bypass = {"127.0.0.1", "localhost", "::1"}
            existing_bypass = child_env.get("NO_PROXY") or child_env.get("no_proxy") or ""
            local_bypass.update(part.strip() for part in existing_bypass.split(",") if part.strip())
            bypass_value = ",".join(sorted(local_bypass))
            child_env["NO_PROXY"] = bypass_value
            child_env["no_proxy"] = bypass_value
            kwargs["env"] = child_env
            if self.platform == "windows":
                kwargs["creationflags"] = (
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "DETACHED_PROCESS", 0)
                )
            else:
                kwargs["start_new_session"] = True

            try:
                process = subprocess.Popen(self.build_command(executable), **kwargs)
            except OSError as exc:
                return {
                    **self.status(),
                    "state": "failed_fallback",
                    "message": f"failed to start Google Chrome: {exc}",
                }
            _STARTED_PROCESS = process
            _STARTED_PLATFORM = self.platform
            _ACTIVE_LEASES = 1

            deadline = time.monotonic() + max(0.1, self.timeout)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    return {
                        **self.status(),
                        "state": "failed_fallback",
                        "message": f"Google Chrome exited with code {process.returncode}",
                    }
                if self.is_cdp_available():
                    return {**self.status(), "state": "ready"}
                time.sleep(0.2)

            return {
                **self.status(),
                "state": "failed_fallback",
                "message": f"Google Chrome CDP did not become ready within {self.timeout:g}s",
            }

    def close_managed(self) -> dict[str, Any]:
        """Release this run's Chrome lease and stop the managed process at zero."""
        global _STARTED_PROCESS, _STARTED_PLATFORM, _ACTIVE_LEASES
        with _START_LOCK:
            process = _STARTED_PROCESS
            if process is None:
                return {**self.status(), "state": "not_managed"}

            if _ACTIVE_LEASES > 0:
                _ACTIVE_LEASES -= 1
            if _ACTIVE_LEASES > 0:
                return {
                    **self.status(),
                    "state": "retained",
                    "active_leases": _ACTIVE_LEASES,
                }

            try:
                if process.poll() is None:
                    self._request_graceful_close()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if self.platform == "windows":
                            process.terminate()
                        else:
                            os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            if self.platform == "windows":
                                process.kill()
                            else:
                                os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=2)
            except (OSError, ProcessLookupError) as exc:
                logger.debug("Managed Chrome was already gone: %s", exc)
            finally:
                _STARTED_PROCESS = None
                _STARTED_PLATFORM = None
                _ACTIVE_LEASES = 0

            return {
                "available": self.is_cdp_available(),
                "platform": self.platform,
                "managed": False,
                "managed_platform": None,
                "cdp_url": self.cdp_url,
                "state": "closed",
            }

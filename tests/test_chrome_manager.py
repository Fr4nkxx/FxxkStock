from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fxxkstock.dataflows.chrome_manager as chrome_manager_module
from fxxkstock.dataflows.chrome_manager import ChromeManager


class FakeStartupInfo:
    def __init__(self):
        self.dwFlags = 0
        self.wShowWindow = None


def config(
    tmp_path: Path,
    platform: str = "ubuntu",
    mode: str = "background",
) -> dict:
    executable = tmp_path / ("chrome.exe" if platform == "windows" else "chrome")
    executable.write_text("", encoding="utf-8")
    return {
        "cn_browser_platform": platform,
        "cn_browser_mode": mode,
        "cn_browser_executable": str(executable),
        "cn_browser_profile_dir": str(tmp_path / "profile"),
        "cn_browser_cdp_url": "http://127.0.0.1:9222",
        "cn_browser_startup_timeout_seconds": 0.01,
    }


def test_build_command_is_local_and_uses_project_profile(tmp_path):
    manager = ChromeManager(config(tmp_path))
    command = manager.build_command(manager.resolve_executable())

    assert "--remote-debugging-address=127.0.0.1" in command
    assert "--remote-debugging-port=9222" in command
    assert f"--user-data-dir={(tmp_path / 'profile').resolve()}" in command
    assert "--start-minimized" in command


def test_headless_command_has_no_window(tmp_path):
    manager = ChromeManager(config(tmp_path, mode="headless"))
    command = manager.build_command(manager.resolve_executable())

    assert "--headless=new" in command
    assert "--disable-gpu" in command
    assert "--window-size=1920,1080" in command
    assert "--start-minimized" not in command


def test_visible_command_has_no_background_flags(tmp_path):
    manager = ChromeManager(config(tmp_path, mode="visible"))
    command = manager.build_command(manager.resolve_executable())

    assert "--headless=new" not in command
    assert "--start-minimized" not in command


def test_invalid_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported Chrome mode"):
        ChromeManager(config(tmp_path, mode="popup"))


def test_existing_cdp_does_not_spawn(tmp_path):
    manager = ChromeManager(config(tmp_path))
    with (
        patch.object(manager, "is_cdp_available", return_value=True),
        patch("fxxkstock.dataflows.chrome_manager.subprocess.Popen") as popen,
    ):
        result = manager.ensure_running()

    assert result["state"] == "already_running"
    popen.assert_not_called()


def test_cdp_health_check_uses_direct_opener(tmp_path):
    manager = ChromeManager(config(tmp_path))
    response = MagicMock()
    response.read.return_value = (
        b'{"webSocketDebuggerUrl":"ws://127.0.0.1:9222/devtools/browser/test"}'
    )
    context = MagicMock()
    context.__enter__.return_value = response
    with patch(
        "fxxkstock.dataflows.chrome_manager._DIRECT_OPENER.open",
        return_value=context,
    ) as direct_open:
        assert manager.is_cdp_available()

    direct_open.assert_called_once_with(
        "http://127.0.0.1:9222/json/version",
        timeout=0.75,
    )


def test_wrong_platform_falls_back_without_spawn(tmp_path):
    manager = ChromeManager(config(tmp_path, platform="windows"))
    with (
        patch.object(manager, "is_cdp_available", return_value=False),
        patch("fxxkstock.dataflows.chrome_manager.current_platform", return_value="macos"),
        patch("fxxkstock.dataflows.chrome_manager.subprocess.Popen") as popen,
    ):
        result = manager.ensure_running()

    assert result["state"] == "failed_fallback"
    assert "does not match" in result["message"]
    popen.assert_not_called()


def test_successful_start_waits_for_cdp(tmp_path):
    manager = ChromeManager(config(tmp_path, platform="macos"))
    process = MagicMock()
    process.poll.return_value = None
    with (
        patch("fxxkstock.dataflows.chrome_manager.current_platform", return_value="macos"),
        patch.object(manager, "is_cdp_available", side_effect=[False, True, True]),
        patch("fxxkstock.dataflows.chrome_manager.subprocess.Popen", return_value=process) as popen,
    ):
        result = manager.ensure_running()

    assert result["state"] == "ready"
    assert popen.call_args.args[0][0].endswith("chrome")
    assert popen.call_args.kwargs["start_new_session"] is True
    assert "127.0.0.1" in popen.call_args.kwargs["env"]["NO_PROXY"]


def test_windows_background_start_does_not_activate_window(tmp_path):
    manager = ChromeManager(config(tmp_path, platform="windows"))
    process = MagicMock()
    process.poll.return_value = None
    with (
        patch("fxxkstock.dataflows.chrome_manager.current_platform", return_value="windows"),
        patch(
            "fxxkstock.dataflows.chrome_manager.subprocess.STARTUPINFO",
            side_effect=FakeStartupInfo,
            create=True,
        ),
        patch(
            "fxxkstock.dataflows.chrome_manager.subprocess.STARTF_USESHOWWINDOW",
            1,
            create=True,
        ),
        patch.object(manager, "is_cdp_available", side_effect=[False, True, True]),
        patch(
            "fxxkstock.dataflows.chrome_manager.subprocess.Popen",
            return_value=process,
        ) as popen,
    ):
        result = manager.ensure_running()

    startupinfo = popen.call_args.kwargs["startupinfo"]
    assert result["state"] == "ready"
    assert "minimized_best_effort" in result["message"]
    assert startupinfo.dwFlags & 1
    assert startupinfo.wShowWindow == chrome_manager_module._SW_SHOWMINNOACTIVE


def test_windows_headless_start_hides_window(tmp_path):
    manager = ChromeManager(config(tmp_path, platform="windows", mode="headless"))
    process = MagicMock()
    process.poll.return_value = None
    with (
        patch("fxxkstock.dataflows.chrome_manager.current_platform", return_value="windows"),
        patch(
            "fxxkstock.dataflows.chrome_manager.subprocess.STARTUPINFO",
            side_effect=FakeStartupInfo,
            create=True,
        ),
        patch(
            "fxxkstock.dataflows.chrome_manager.subprocess.STARTF_USESHOWWINDOW",
            1,
            create=True,
        ),
        patch.object(manager, "is_cdp_available", side_effect=[False, True, True]),
        patch(
            "fxxkstock.dataflows.chrome_manager.subprocess.Popen",
            return_value=process,
        ) as popen,
    ):
        result = manager.ensure_running()

    assert result["state"] == "ready"
    assert popen.call_args.kwargs["startupinfo"].wShowWindow == chrome_manager_module._SW_HIDE


def test_concurrent_calls_spawn_once(tmp_path):
    manager = ChromeManager(config(tmp_path))
    process = MagicMock()
    process.poll.return_value = None
    available = iter([False, True, True, True, True])
    results = []

    with (
        patch("fxxkstock.dataflows.chrome_manager.current_platform", return_value="ubuntu"),
        patch.object(
            manager, "is_cdp_available", side_effect=lambda *args, **kwargs: next(available, True)
        ),
        patch("fxxkstock.dataflows.chrome_manager.subprocess.Popen", return_value=process) as popen,
    ):
        threads = [
            threading.Thread(target=lambda: results.append(manager.ensure_running()))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert popen.call_count == 1
    assert {result["state"] for result in results} <= {"ready", "already_running"}


def test_status_includes_profile_and_lease_state(tmp_path):
    manager = ChromeManager(config(tmp_path))
    with patch.object(manager, "is_cdp_available", return_value=False):
        status = manager.status()

    assert status["profile_dir"] == str((tmp_path / "profile").resolve())
    assert status["profile_exists"] is False
    assert status["active_leases"] == 0
    assert status["can_close"] is False
    assert status["mode"] == "background"
    assert status["managed_mode"] is None
    assert status["window_behavior"] == "minimized_best_effort"


def test_headless_status_reports_no_window(tmp_path):
    manager = ChromeManager(config(tmp_path, mode="headless"))
    with patch.object(manager, "is_cdp_available", return_value=False):
        status = manager.status()

    assert status["window_behavior"] == "no_window"


def test_close_requests_graceful_browser_shutdown_first(tmp_path, monkeypatch):
    manager = ChromeManager(config(tmp_path))
    process = MagicMock()
    process.poll.return_value = None
    monkeypatch.setattr(chrome_manager_module, "_STARTED_PROCESS", process)
    monkeypatch.setattr(chrome_manager_module, "_STARTED_PLATFORM", "ubuntu")
    monkeypatch.setattr(chrome_manager_module, "_STARTED_MODE", "background")
    monkeypatch.setattr(chrome_manager_module, "_ACTIVE_LEASES", 1)

    with (
        patch.object(manager, "_request_graceful_close", return_value=True) as graceful,
        patch.object(manager, "is_cdp_available", return_value=False),
    ):
        result = manager.close_managed()

    graceful.assert_called_once()
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    assert result["state"] == "closed"
    assert result["managed_mode"] is None

"""Tests for pi_evals.cli.

Monkeypatches pytest.main itself (the actual installed pytest module, which
cli.main() imports and calls at runtime) so these tests exercise the CLI's
own orchestration — env var wiring, phase sequencing, exit-code
short-circuiting — without spawning a real nested pytest session.
"""

from __future__ import annotations

import pytest as pytest_module

from pi_evals import cli


class _FakeMain:
    def __init__(self, exit_codes: list[int]) -> None:
        self._exit_codes = list(exit_codes)
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> int:
        self.calls.append(list(args))
        return self._exit_codes.pop(0)


class TestCliMain:
    def test_runs_eval_then_analysis_phase(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        fake = _FakeMain([0, 0])
        monkeypatch.setattr(pytest_module, "main", fake)

        exit_code = cli.main([])
        assert exit_code == 0
        assert len(fake.calls) == 2
        assert "--run-eval" in fake.calls[0]
        assert "--supress-failed-exit-code" in fake.calls[0]
        assert "--run-eval-analysis" in fake.calls[1]

    def test_analysis_phase_exit_code_is_returned(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        fake = _FakeMain([0, 3])
        monkeypatch.setattr(pytest_module, "main", fake)
        assert cli.main([]) == 3

    def test_eval_phase_error_skips_analysis_phase(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        fake = _FakeMain([2])
        monkeypatch.setattr(pytest_module, "main", fake)
        exit_code = cli.main([])
        assert exit_code == 2
        assert len(fake.calls) == 1

    def test_provider_and_model_set_env_vars(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
        monkeypatch.delenv("PI_MODEL", raising=False)
        fake = _FakeMain([0, 0])
        monkeypatch.setattr(pytest_module, "main", fake)

        cli.main(["--provider", "openai", "--model", "gpt-5"])
        import os

        assert os.environ["PI_PROVIDER"] == "openai"
        assert os.environ["PI_MODEL"] == "gpt-5"

    def test_extra_args_forwarded_to_both_phases(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        fake = _FakeMain([0, 0])
        monkeypatch.setattr(pytest_module, "main", fake)

        cli.main(["-k", "smoke"])
        assert "-k" in fake.calls[0]
        assert "smoke" in fake.calls[0]
        assert "-k" in fake.calls[1]
        assert "smoke" in fake.calls[1]

    def test_provider_without_model_errors(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        with pytest_module.raises(SystemExit):
            cli.main(["--provider", "openai"])

    def test_model_without_provider_errors(self, monkeypatch: pytest_module.MonkeyPatch) -> None:
        with pytest_module.raises(SystemExit):
            cli.main(["--model", "gpt-5"])

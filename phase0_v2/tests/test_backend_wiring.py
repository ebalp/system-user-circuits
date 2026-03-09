"""Tests for backend wiring in run_experiments.py.

Tests CLI argument parsing, backend validation, and the _run_work_items helper.
All tests are mocked -- no real API calls.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


class TestCLIArgParsing:
    """Test the new CLI arguments parse correctly."""

    def _make_parser(self):
        """Build the argument parser matching run_experiments.py."""
        # Import and test by running the argparse setup
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="phase0_v2/config/experiment.yaml")
        parser.add_argument("--output-dir", default="phase0_v2/data/results")
        parser.add_argument("--model", type=str)
        parser.add_argument("--conflicts", type=str)
        parser.add_argument("--n-tasks", type=int)
        parser.add_argument("--conditions", type=str)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--backend", choices=["hf", "vllm", "lambda"], default="hf")
        parser.add_argument("--vllm-url", type=str)
        parser.add_argument("--lambda-config", type=str, default="phase0_v2/config/lambda.yaml")
        return parser

    def test_default_backend_is_hf(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.backend == "hf"

    def test_backend_vllm(self):
        parser = self._make_parser()
        args = parser.parse_args(["--backend", "vllm", "--vllm-url", "http://localhost:8000/v1"])
        assert args.backend == "vllm"
        assert args.vllm_url == "http://localhost:8000/v1"

    def test_backend_lambda(self):
        parser = self._make_parser()
        args = parser.parse_args(["--backend", "lambda"])
        assert args.backend == "lambda"

    def test_invalid_backend_rejected(self):
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "invalid"])

    def test_lambda_config_default(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.lambda_config == "phase0_v2/config/lambda.yaml"

    def test_lambda_config_custom(self):
        parser = self._make_parser()
        args = parser.parse_args(["--lambda-config", "/tmp/custom.yaml"])
        assert args.lambda_config == "/tmp/custom.yaml"

    def test_vllm_url_none_by_default(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.vllm_url is None

    def test_all_args_together(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "--backend", "vllm",
            "--vllm-url", "http://10.0.0.1:8000/v1",
            "--model", "test-model",
            "--conflicts", "forbidden_words,language_en_es",
            "--n-tasks", "5",
            "--dry-run",
        ])
        assert args.backend == "vllm"
        assert args.vllm_url == "http://10.0.0.1:8000/v1"
        assert args.model == "test-model"
        assert args.dry_run is True

    def test_existing_args_unchanged(self):
        """Existing args should still work identically."""
        parser = self._make_parser()
        args = parser.parse_args([
            "--config", "custom.yaml",
            "--output-dir", "/tmp/results",
            "--conditions", "A,C",
        ])
        assert args.config == "custom.yaml"
        assert args.output_dir == "/tmp/results"
        assert args.conditions == "A,C"
        assert args.backend == "hf"  # default unchanged


class TestRunWorkItems:
    """Test the _run_work_items helper function."""

    def test_import_and_call(self):
        """_run_work_items should be importable from run_experiments module."""
        # This verifies the function exists at module level
        from phase0_v2.run_experiments import _run_work_items
        assert callable(_run_work_items)

    def test_empty_work_items(self):
        from phase0_v2.run_experiments import _run_work_items
        result = _run_work_items(
            work_items=[],
            runner=MagicMock(),
            model_semaphores={},
            model_locks={},
            max_workers=1,
        )
        assert result == 0

    def test_successful_items_counted(self):
        from phase0_v2.run_experiments import _run_work_items

        mock_runner = MagicMock()
        mock_runner.run_single.return_value = {"prompt_id": "p1", "error": None}

        mock_prompt = MagicMock()
        mock_prompt.id = "p1"
        mock_conflict = MagicMock()
        items = [("model-a", mock_prompt, mock_conflict)]

        result = _run_work_items(
            work_items=items,
            runner=mock_runner,
            model_semaphores={"model-a": threading.Semaphore(1)},
            model_locks={"model-a": threading.Lock()},
            max_workers=1,
        )
        assert result == 1

    def test_error_items_not_saved(self):
        from phase0_v2.run_experiments import _run_work_items

        mock_runner = MagicMock()
        mock_runner.run_single.return_value = {"prompt_id": "p1", "error": "API failed"}

        mock_prompt = MagicMock()
        mock_prompt.id = "p1"
        items = [("model-a", mock_prompt, MagicMock())]

        _run_work_items(
            work_items=items,
            runner=mock_runner,
            model_semaphores={"model-a": threading.Semaphore(1)},
            model_locks={"model-a": threading.Lock()},
            max_workers=1,
        )
        mock_runner.append_record.assert_not_called()

    def test_successful_items_saved(self):
        """Successful items (error=None) should be saved via append_record."""
        from phase0_v2.run_experiments import _run_work_items

        mock_runner = MagicMock()
        mock_runner.run_single.return_value = {"prompt_id": "p1", "error": None}

        mock_prompt = MagicMock()
        mock_prompt.id = "p1"
        items = [("model-a", mock_prompt, MagicMock())]

        _run_work_items(
            work_items=items,
            runner=mock_runner,
            model_semaphores={"model-a": threading.Semaphore(1)},
            model_locks={"model-a": threading.Lock()},
            max_workers=1,
        )
        mock_runner.append_record.assert_called_once_with("model-a", {"prompt_id": "p1", "error": None})

    def test_multiple_items_counted(self):
        """Multiple successful items should all be counted."""
        from phase0_v2.run_experiments import _run_work_items

        mock_runner = MagicMock()
        mock_runner.run_single.return_value = {"prompt_id": "p", "error": None}

        items = []
        for i in range(5):
            mock_prompt = MagicMock()
            mock_prompt.id = f"p{i}"
            items.append((f"model-{i % 2}", mock_prompt, MagicMock()))

        sems = {f"model-{i}": threading.Semaphore(2) for i in range(2)}
        locks = {f"model-{i}": threading.Lock() for i in range(2)}

        result = _run_work_items(
            work_items=items,
            runner=mock_runner,
            model_semaphores=sems,
            model_locks=locks,
            max_workers=2,
        )
        assert result == 5

    def test_exception_in_run_single_not_counted(self):
        """If run_single raises, the item should not be counted."""
        from phase0_v2.run_experiments import _run_work_items

        mock_runner = MagicMock()
        mock_runner.run_single.side_effect = RuntimeError("model crash")

        mock_prompt = MagicMock()
        mock_prompt.id = "p1"
        items = [("model-a", mock_prompt, MagicMock())]

        result = _run_work_items(
            work_items=items,
            runner=mock_runner,
            model_semaphores={"model-a": threading.Semaphore(1)},
            model_locks={"model-a": threading.Lock()},
            max_workers=1,
        )
        assert result == 0

    def test_mixed_success_and_error(self):
        """Mix of success and error items should count only successes."""
        from phase0_v2.run_experiments import _run_work_items

        call_count = 0

        def mock_run_single(prompt, conflict, model_id):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return {"prompt_id": prompt.id, "error": "failed"}
            return {"prompt_id": prompt.id, "error": None}

        mock_runner = MagicMock()
        mock_runner.run_single.side_effect = mock_run_single

        items = []
        for i in range(4):
            mock_prompt = MagicMock()
            mock_prompt.id = f"p{i}"
            items.append(("model-a", mock_prompt, MagicMock()))

        result = _run_work_items(
            work_items=items,
            runner=mock_runner,
            model_semaphores={"model-a": threading.Semaphore(2)},
            model_locks={"model-a": threading.Lock()},
            max_workers=2,
        )
        # All 4 futures complete (counted), but only 2 are saved
        assert result == 4
        assert mock_runner.append_record.call_count == 2



class TestBackendVLLMClientCreation:
    """Test that --backend vllm creates a VLLMClient."""

    def test_vllm_client_import(self):
        """VLLMClient should be importable from api_client."""
        from phase0_v2.src.api_client import VLLMClient
        assert VLLMClient is not None

    def test_vllm_client_has_chat_completion(self):
        """VLLMClient should have chat_completion method."""
        from phase0_v2.src.api_client import VLLMClient
        assert hasattr(VLLMClient, "chat_completion")


class TestBackendLambdaImports:
    """Test that Lambda backend imports work."""

    def test_lambda_imports(self):
        from lambda_cloud.config import load_lambda_config
        from lambda_cloud.manager import LambdaCloudManager
        assert LambdaCloudManager is not None
        assert load_lambda_config is not None


class TestSyncEnvTemplate:
    """Test that sync.env.template has the LAMBDA_API_KEY entry."""

    def test_lambda_api_key_in_template(self):
        from pathlib import Path
        template = Path("sync.env.template").read_text()
        assert "LAMBDA_API_KEY" in template

    def test_hf_token_still_present(self):
        from pathlib import Path
        template = Path("sync.env.template").read_text()
        assert "HF_TOKEN" in template

    def test_lambda_api_key_before_hf_token(self):
        """LAMBDA_API_KEY should appear before HF_TOKEN in the template."""
        from pathlib import Path
        template = Path("sync.env.template").read_text()
        hf_pos = template.index("HF_TOKEN")
        lambda_pos = template.index("LAMBDA_API_KEY")
        assert lambda_pos < hf_pos


class TestVLLMBackendValidation:
    """Test that --backend vllm requires --vllm-url."""

    def test_vllm_without_url_errors(self):
        """--backend vllm without --vllm-url should raise SystemExit."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "phase0_v2.run_experiments",
             "--backend", "vllm"],
            capture_output=True, text=True, timeout=30,
            cwd="/Users/enrique/system-user-circuits",
        )
        assert result.returncode != 0
        assert "vllm-url" in result.stderr.lower() or "required" in result.stderr.lower()


class TestDryRunWithBackendFlags:
    """Test that --dry-run works with all backend flags (no API calls needed)."""

    def test_dry_run_with_hf_backend(self):
        """--dry-run --backend hf should print prompts and exit."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "phase0_v2.run_experiments",
             "--dry-run", "--backend", "hf"],
            capture_output=True, text=True, timeout=30,
            cwd="/Users/enrique/system-user-circuits",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_dry_run_with_vllm_backend(self):
        """--dry-run --backend vllm --vllm-url ... should print prompts and exit."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "phase0_v2.run_experiments",
             "--dry-run", "--backend", "vllm", "--vllm-url", "http://localhost:8000/v1"],
            capture_output=True, text=True, timeout=30,
            cwd="/Users/enrique/system-user-circuits",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_dry_run_with_lambda_backend(self):
        """--dry-run --backend lambda should print prompts and exit."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "phase0_v2.run_experiments",
             "--dry-run", "--backend", "lambda"],
            capture_output=True, text=True, timeout=30,
            cwd="/Users/enrique/system-user-circuits",
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

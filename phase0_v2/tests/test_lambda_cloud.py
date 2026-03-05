"""Tests that the phase0_v2.src.lambda_cloud shim re-exports correctly.

Full test coverage lives in lambda_cloud/tests/. This file only verifies
that the backward-compatible imports still work.
"""

from phase0_v2.src.lambda_cloud import (
    LambdaConfig,
    LambdaInstance,
    LambdaCloudManager,
    load_lambda_config,
)

from lambda_cloud.config import LambdaConfig as _LambdaConfig
from lambda_cloud.config import LambdaInstance as _LambdaInstance
from lambda_cloud.config import load_lambda_config as _load_lambda_config
from lambda_cloud.manager import LambdaCloudManager as _LambdaCloudManager


def test_shim_reexports_lambdaconfig():
    assert LambdaConfig is _LambdaConfig


def test_shim_reexports_lambdainstance():
    assert LambdaInstance is _LambdaInstance


def test_shim_reexports_lambdacloudmanager():
    assert LambdaCloudManager is _LambdaCloudManager


def test_shim_reexports_load_lambda_config():
    assert load_lambda_config is _load_lambda_config

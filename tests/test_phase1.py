from __future__ import annotations

import pathlib
import tomllib


def test_pyproject_is_valid_toml() -> None:
    pyproject = pathlib.Path("pyproject.toml")
    assert pyproject.exists()
    data = tomllib.loads(pyproject.read_text())
    assert "project" in data
    assert data["project"]["name"] == "kenya-sme-financial-intelligence"


def test_settings_module_exists() -> None:
    settings_file = pathlib.Path("src/config/settings.py")
    assert settings_file.exists()


def test_env_example_exists() -> None:
    assert pathlib.Path(".env.example").exists()


def test_docker_compose_exists() -> None:
    assert pathlib.Path("docker-compose.yml").exists()
    assert pathlib.Path("Dockerfile").exists()

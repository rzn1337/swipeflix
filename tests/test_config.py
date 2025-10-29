"""Tests for configuration management."""

import os

import pytest


def test_settings_default_values():
    """Test default settings values."""
    from swipeflix.config import Settings

    settings = Settings()

    assert settings.app_name == "SwipeFlix"
    assert settings.api_port == 8000
    assert settings.random_seed == 42
    assert settings.n_components == 50


def test_settings_from_env(monkeypatch):
    """Test settings from environment variables."""
    from swipeflix.config import Settings

    monkeypatch.setenv("APP_NAME", "TestApp")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings()

    assert settings.app_name == "TestApp"
    assert settings.api_port == 9000
    assert settings.debug is True


def test_settings_model_config():
    """Test model configuration settings."""
    from swipeflix.config import Settings

    settings = Settings()

    assert settings.model_name == "SwipeFlixModel"
    assert settings.model_version == "1"
    assert settings.content_weight + settings.collab_weight == 1.0


def test_settings_paths():
    """Test data path settings."""
    from swipeflix.config import Settings

    settings = Settings()

    assert settings.movies_file == "movies.csv"
    assert settings.ratings_file == "ratings.csv"
    assert settings.data_dir.name == "data"


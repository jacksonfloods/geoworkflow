# File: src/geoworkflow/cli/__init__.py
"""
Command-line interface for the geoworkflow package.

This module provides a unified CLI that replaces the individual scripts
with a single entry point and multiple subcommands.
"""

from geoworkflow.cli.main import main

__all__ = ["main"]


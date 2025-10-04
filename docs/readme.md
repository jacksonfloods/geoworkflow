# GeoWorkflow Documentation

This directory contains all documentation for the GeoWorkflow project.

## Documentation Structure

- **`api/`** - API Reference (Sphinx)
  - Auto-generated from Python docstrings
  - Technical reference for developers
  - Build: `cd api && make html`
  - Output: `api/build/html/`

- **`guide/`** - User Guide (MkDocs)
  - Tutorials, how-tos, and conceptual guides
  - User-focused documentation
  - Build: `mkdocs build` (from project root)
  - Preview: `mkdocs serve`

## Building Documentation

### API Reference (Sphinx)
```bash
cd docs/api
make html
# Open docs/api/build/html/index.html

# Contributing to GeoWorkflow

We welcome contributions! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/jacksonfloods/geoworkflow.git
cd geoworkflow

# Create development environment
conda env create -f environment.yml
conda activate geoworkflow

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Code Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use Google-style docstrings
- Type hints for function signatures

### Running Formatters

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check types
mypy src/geoworkflow

# Lint
flake8 src/ tests/
```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/ -v
   pytest --cov=geoworkflow --cov-report=html
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-new-feature
   ```

## Testing Guidelines

- Write tests for all new processors
- Aim for >80% code coverage
- Use pytest fixtures for common test data
- Test both success and failure cases

Example test:

```python
import pytest
from geoworkflow.processors.spatial.clipper import ClippingProcessor

def test_clipper_basic(tmp_path):
    """Test basic clipping functionality."""
    config = {
        "input_dir": "tests/fixtures/rasters",
        "output_dir": tmp_path,
        "boundary_file": "tests/fixtures/boundaries/test_aoi.geojson"
    }
    
    processor = ClippingProcessor(config)
    result = processor.process()
    
    assert result.success
    assert result.processed_count > 0
```

## Documentation

- Update docstrings for any changed functions
- Add examples to documentation files
- Update README if adding major features
- Build docs locally to check formatting:
  ```bash
  cd docs
  make html
  mkdocs serve
  ```

## Pull Request Process

1. Ensure all tests pass
2. Update CHANGELOG.md
3. Request review from maintainers
4. Address review feedback
5. Squash commits if requested

## Questions?

Open an issue or reach out to the maintainers.

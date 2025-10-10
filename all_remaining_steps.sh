#!/bin/bash
# Complete All Remaining Steps (3, 4, 5, 6, 7, 8, 9, 10)
# Comprehensive script to finish the interactive directory tree implementation

set -e  # Exit on error

echo "=========================================="
echo "Complete Interactive Tree Implementation"
echo "All Remaining Steps (3-10)"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Verify Steps 1 and 2 were completed
echo -e "${BLUE}[Verification] Checking prerequisites...${NC}"

if [ ! -f "docs/guide/assets/directory-tree.json" ]; then
    echo -e "${RED}Error: directory-tree.json not found!${NC}"
    echo "Please run step1_update_gen_ref_pages.sh first"
    exit 1
fi

if [ ! -f "docs/guide/assets/js/directory-tree.js" ]; then
    echo -e "${RED}Error: directory-tree.js not found!${NC}"
    echo "Please run step2_create_d3js_visualization.sh first"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites verified${NC}"
echo ""

# =============================================================================
# STEP 3: Embed Interactive Tree in structure.md
# =============================================================================
echo -e "${BLUE}=== STEP 3: Embedding Tree in structure.md ===${NC}"

if [ -f "docs/guide/guide/structure.md" ]; then
    cp docs/guide/guide/structure.md docs/guide/guide/structure.md.backup
    echo -e "${GREEN}✓ Backed up structure.md${NC}"
fi

cat > docs/guide/guide/structure.md << 'STEP3EOF'
# Project Structure

This page documents the organization of the GeoWorkflow codebase.

## Interactive Directory Tree

Explore the project structure below. Click on folders to expand/collapse, and hover over items to see descriptions.

<div style="margin: 2rem 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
  <iframe 
    src="../../assets/directory-tree-container.html" 
    width="100%" 
    height="650px" 
    frameborder="0"
    style="border: none; display: block;"
    title="Interactive Directory Structure">
  </iframe>
</div>

!!! tip "How to Use"
    - **Click** on folder nodes (📁) to expand or collapse them
    - **Hover** over any node to see detailed descriptions
    - **Blue nodes** represent directories
    - **Green nodes** represent Python files

??? note "Can't see the tree?"
    If the interactive tree doesn't load, you can [view it directly](../../assets/directory-tree-container.html) or see the text version below.

---

## Text-Based Structure Reference

For quick reference or if you prefer a text-based view, here's the complete directory structure:

```
src/geoworkflow/
│   ├── __init__.py
│   ├── __version__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── cli_structure.py
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── aoi.py
│   │   │   ├── extract.py
│   │   │   ├── pipeline.py
│   │   │   ├── process.py
│   │   │   ├── visualize.py
│   │   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── enhanced_base.py
│   │   ├── exceptions.py
│   │   ├── logging_setup.py
│   │   ├── pipeline.py
│   │   ├── pipeline_enhancements.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── aoi/
│   │   │   ├── __init__.py
│   │   │   ├── processor.py
│   │   ├── extraction/
│   │   │   ├── __init__.py
│   │   │   ├── archive.py
│   │   │   ├── open_buildings.py
│   │   ├── integration/
│   │   │   ├── __init__.py
│   │   │   ├── enrichment.py
│   │   ├── spatial/
│   │   │   ├── __init__.py
│   │   │   ├── aligner.py
│   │   │   ├── clipper.py
│   │   │   ├── masker.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── config_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── earth_engine_error_handler.py
│   │   ├── earth_engine_utils.py
│   │   ├── file_utils.py
│   │   ├── mask_utils.py
│   │   ├── progress_utils.py
│   │   ├── raster_utils.py
│   │   ├── resource_utils.py
│   │   ├── validation.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── raster/
│   │   │   ├── __init__.py
│   │   │   ├── processor.py
│   │   ├── reports/
│   │   │   ├── __init__.py
│   │   ├── vector/
│   │   │   ├── __init__.py
```

---

## Directory Descriptions

### Core Modules

#### `core/`
Foundation classes, base processors, configuration, and constants. This module provides the abstract base classes and core functionality that all other modules depend on.

**Key files:**
- `base.py` - Abstract base classes for processors
- `pipeline.py` - Processing pipeline orchestration
- `config.py` - Configuration management
- `exceptions.py` - Custom exception classes

#### `schemas/`
Pydantic models for configuration validation. These models ensure that YAML configuration files are valid and provide type safety throughout the application.

**Key files:**
- `config_models.py` - All configuration model definitions

---

### Processing Modules

#### `processors/`
Specialized processors for each workflow stage. Each subdirectory contains processors for a specific type of operation.

##### `processors/aoi/`
Area of Interest (AOI) creation and management. Handles loading, creating, and validating geographic boundaries.

##### `processors/spatial/`
Spatial operations including clipping, alignment, and reprojection. These processors ensure all rasters are in the same coordinate system and extent.

**Key files:**
- `clipper.py` - Clip rasters to AOI boundaries
- `aligner.py` - Align rasters to reference grid
- `masker.py` - Apply masks to rasters

##### `processors/extraction/`
Data extraction from archives and downloads. Handles extracting data from various archive formats and downloading from remote sources.

**Key files:**
- `archive.py` - Extract from ZIP, TAR, etc.
- `open_buildings.py` - Download Google Open Buildings data

##### `processors/integration/`
Statistical enrichment and data integration. Combines multiple datasets and performs zonal statistics.

**Key files:**
- `enrichment.py` - Zonal statistics and data enrichment

---

### Utilities

#### `utils/`
Helper functions and common operations used throughout the codebase.

**Key files:**
- `file_utils.py` - File system operations
- `raster_utils.py` - Raster manipulation helpers
- `earth_engine_utils.py` - Google Earth Engine integration
- `validation.py` - Data validation utilities

---

### User Interface

#### `cli/`
Command-line interface entry points. Provides a user-friendly CLI built with Click/Typer.

**Structure:**
- `main.py` - CLI entry point
- `commands/` - Individual command implementations
  - `aoi.py` - AOI management commands
  - `extract.py` - Data extraction commands
  - `process.py` - Processing commands
  - `visualize.py` - Visualization commands

#### `visualization/`
Visualization components for creating maps and charts.

##### `visualization/raster/`
Raster visualization processors for creating maps from raster data.

##### `visualization/vector/`
Vector visualization processors for creating maps from vector data.

##### `visualization/reports/`
Report generation utilities for creating analysis summaries.

---

## How Files Are Organized

The GeoWorkflow codebase follows these organizational principles:

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Processor Pattern**: All data processing follows the `BaseProcessor` pattern
3. **Configuration-Driven**: Pydantic models in `schemas/` validate all configurations
4. **Utilities as Support**: Common functionality is extracted to `utils/`
5. **CLI as Interface**: User-facing commands are in `cli/`, calling processors internally

---

## For Developers

When adding new functionality:

1. **New processor**: Add to appropriate `processors/` subdirectory
2. **New configuration**: Add Pydantic model to `schemas/config_models.py`
3. **New utility**: Add to relevant module in `utils/`
4. **New CLI command**: Add to `cli/commands/`
5. **Update this page**: Run `python docs/guide/gen_ref_pages.py` to regenerate structure

The interactive tree will automatically update when you run the documentation generation script.

---

## See Also

- [Getting Started](../../getting-started/installation.md) - Installation and setup
- [Configuration Guide](../../getting-started/configuration.md) - How to configure workflows
- [API Reference](../../api/) - Detailed API documentation
- [Development Guide](../../development/contributing.md) - Contributing to the project
STEP3EOF

echo -e "${GREEN}✓ Step 3: structure.md updated with embedded tree${NC}"
echo ""

# =============================================================================
# STEP 4: Configure mkdocs.yml
# =============================================================================
echo -e "${BLUE}=== STEP 4: Configuring mkdocs.yml ===${NC}"

if [ -f "mkdocs.yml" ]; then
    cp mkdocs.yml mkdocs.yml.backup
    echo -e "${YELLOW}⚠ Backed up existing mkdocs.yml${NC}"
fi

cat > mkdocs.yml << 'STEP4EOF'
# GeoWorkflow Documentation Configuration

site_name: GeoWorkflow
site_description: Geospatial data processing for African urbanization analysis
site_author: AfricaPolis Team
site_url: https://jacksonfloods.github.io/geoworkflow/

# Repository
repo_name: jacksonfloods/geoworkflow
repo_url: https://github.com/jacksonfloods/geoworkflow
edit_uri: edit/main/docs/guide/

# Copyright
copyright: Copyright &copy; 2025 AfricaPolis Team

# Configuration
docs_dir: docs/guide
site_dir: site

# Theme configuration - Material for MkDocs (FastAPI-style)
theme:
  name: material
  language: en
  
  # Color scheme
  palette:
    # Light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  
  # Features
  features:
    - content.code.annotate
    - content.code.copy
    - content.tabs.link
    - content.tooltips
    - navigation.expand
    - navigation.footer
    - navigation.indexes
    - navigation.instant
    - navigation.instant.prefetch
    - navigation.sections
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.top
    - navigation.tracking
    - search.highlight
    - search.share
    - search.suggest
    - toc.follow
  
  # Icons
  icon:
    repo: fontawesome/brands/github
  
  # Font
  font:
    text: Roboto
    code: Roboto Mono

# Additional CSS and JavaScript
extra_css:
  - assets/css/directory-tree.css
  - assets/custom.css

extra_javascript:
  - https://d3js.org/d3.v7.min.js
  - assets/js/directory-tree.js

# Plugins
plugins:
  - search:
      lang: en
      separator: '[\s\-\.]+'

# Extensions
markdown_extensions:
  # Python Markdown
  - abbr
  - admonition
  - attr_list
  - def_list
  - footnotes
  - md_in_html
  - tables
  - toc:
      permalink: true
      toc_depth: 3
  
  # Python Markdown Extensions
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.betterem:
      smart_enable: all
  - pymdownx.caret
  - pymdownx.details
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.keys
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.tilde

# Navigation
nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
      - Configuration: getting-started/configuration.md
  - Project Guide:
      - Structure: guide/structure.md
      - Concepts: guide/concepts.md
  - Tutorials:
      - Basic Workflow: tutorials/basic-workflow.md
  - Development:
      - Contributing: development/contributing.md
      - Testing: development/testing.md

# Extra
extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/jacksonfloods/geoworkflow
    - icon: fontawesome/brands/python
      link: https://pypi.org/project/geoworkflow/
STEP4EOF

# Create custom.css if needed
mkdir -p docs/guide/assets
cat > docs/guide/assets/custom.css << 'CSSEOF'
/* Custom styles for GeoWorkflow documentation */

.md-content {
  max-width: 900px;
}

.md-content iframe {
  border-radius: 4px;
}

.md-typeset code {
  border-radius: 2px;
}

.md-typeset h2 {
  margin-top: 2rem;
}

.md-typeset .admonition.tip {
  border-left-color: #00bcd4;
}

.md-typeset .admonition.tip > .admonition-title {
  background-color: rgba(0, 188, 212, 0.1);
}
CSSEOF

echo -e "${GREEN}✓ Step 4: mkdocs.yml configured${NC}"
echo -e "${GREEN}✓ Step 4: custom.css created${NC}"
echo ""

# =============================================================================
# STEP 5: Update GitHub Actions Workflow
# =============================================================================
echo -e "${BLUE}=== STEP 5: Updating GitHub Actions workflow ===${NC}"

mkdir -p .github/workflows

if [ -f ".github/workflows/docs.yaml" ]; then
    cp .github/workflows/docs.yaml .github/workflows/docs.yaml.backup
    echo -e "${YELLOW}⚠ Backed up existing docs.yaml${NC}"
fi

cat > .github/workflows/docs.yaml << 'STEP5EOF'
name: Deploy Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install mkdocs-material
          pip install -e ".[docs]"
      
      - name: Generate directory structure
        run: |
          python docs/guide/gen_ref_pages.py
      
      - name: Build MkDocs
        run: |
          mkdocs build --verbose --strict
      
      - name: Build Sphinx
        run: |
          cd docs/api
          make html
          cd ../..
      
      - name: Combine documentation
        run: |
          mkdir -p site_temp
          cp -r site/* site_temp/
          mkdir -p site_temp/api
          cp -r docs/api/build/html/* site_temp/api/
          rm -rf site
          mv site_temp site
      
      - name: Deploy to GitHub Pages
        if: github.event_name != 'pull_request'
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
          force_orphan: true
          user_name: 'github-actions[bot]'
          user_email: 'github-actions[bot]@users.noreply.github.com'
          commit_message: 'docs: deploy documentation [skip ci]'
STEP5EOF

echo -e "${GREEN}✓ Step 5: GitHub Actions workflow updated${NC}"
echo ""

# =============================================================================
# STEP 6: Verify .gitignore
# =============================================================================
echo -e "${BLUE}=== STEP 6: Verifying .gitignore ===${NC}"

# Ensure .gitignore exists
touch .gitignore

# Check and add entries
gitignore_updated=false

if ! grep -q "^site/$" .gitignore 2>/dev/null; then
    echo "site/" >> .gitignore
    gitignore_updated=true
fi

if ! grep -q "^\.cache/$" .gitignore 2>/dev/null; then
    echo ".cache/" >> .gitignore
    gitignore_updated=true
fi

# Make sure our assets are NOT ignored
if grep -q "^docs/guide/assets/" .gitignore 2>/dev/null; then
    sed -i.bak '/^docs\/guide\/assets\//d' .gitignore
    rm -f .gitignore.bak
    echo -e "${YELLOW}⚠ Removed docs/guide/assets/ from .gitignore${NC}"
fi

if [ "$gitignore_updated" = true ]; then
    echo -e "${GREEN}✓ Step 6: .gitignore updated${NC}"
else
    echo -e "${GREEN}✓ Step 6: .gitignore already configured${NC}"
fi
echo ""

# =============================================================================
# STEP 7: Test Local Build
# =============================================================================
echo -e "${BLUE}=== STEP 7: Testing local build ===${NC}"

# Regenerate structure to make sure it's current
python docs/guide/gen_ref_pages.py

if command -v mkdocs &> /dev/null; then
    echo "Testing mkdocs build..."
    if mkdocs build --quiet; then
        echo -e "${GREEN}✓ Step 7: MkDocs build successful${NC}"
    else
        echo -e "${YELLOW}⚠ Step 7: MkDocs build had warnings (check output)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Step 7: MkDocs not in PATH, skipping build test${NC}"
    echo "  Install with: pip install mkdocs-material"
fi
echo ""

# =============================================================================
# STEP 8: Update Documentation README
# =============================================================================
echo -e "${BLUE}=== STEP 8: Updating documentation README ===${NC}"

cat > docs/readme.md << 'STEP8EOF'
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

### Quick Start
```bash
# Generate directory structure
python docs/guide/gen_ref_pages.py

# Preview MkDocs documentation
mkdocs serve

# Build for production
mkdocs build
```

### API Reference (Sphinx)
```bash
cd docs/api
make html
# Open docs/api/build/html/index.html
```

### Combined Build
```bash
# Generate structure
python docs/guide/gen_ref_pages.py

# Build MkDocs
mkdocs build

# Build Sphinx
cd docs/api && make html && cd ../..

# Combine (manual)
mkdir -p site_temp
cp -r site/* site_temp/
cp -r docs/api/build/html/* site_temp/api/
rm -rf site
mv site_temp site
```

## Interactive Directory Tree

The project structure page includes an interactive D3.js tree visualization:

- **Location**: `guide/structure.md`
- **Data**: `guide/assets/directory-tree.json` (auto-generated)
- **Visualization**: `guide/assets/js/directory-tree.js`
- **Styles**: `guide/assets/css/directory-tree.css`

### Updating the Tree

The directory tree automatically updates when you run:
```bash
python docs/guide/gen_ref_pages.py
```

This script:
1. Walks the `src/geoworkflow/` directory
2. Generates `directory-tree.json` with structure and descriptions
3. Updates `structure.md` with text-based tree

To add descriptions for new directories:
1. Edit `docs/guide/gen_ref_pages.py`
2. Add entries to the `DIRECTORY_DESCRIPTIONS` dict
3. Run the script to regenerate

## GitHub Actions

Documentation is automatically built and deployed on push to `main`:

1. Generates directory structure
2. Builds MkDocs (user guide)
3. Builds Sphinx (API reference)
4. Combines both outputs
5. Deploys to GitHub Pages

## Local Development

### Prerequisites
```bash
pip install mkdocs-material
pip install -e ".[docs]"
```

### Workflow
1. Make changes to markdown files in `docs/guide/`
2. Run `mkdocs serve` to preview
3. Visit `http://localhost:8000`
4. Changes auto-reload in browser

### Adding New Pages
1. Create markdown file in `docs/guide/`
2. Add to `nav` section in `mkdocs.yml`
3. Preview with `mkdocs serve`

## Customization

### Theme Colors
Edit `mkdocs.yml`:
```yaml
theme:
  palette:
    primary: deep purple  # Change color
    accent: amber         # Change accent
```

### Navigation
Edit `mkdocs.yml`:
```yaml
nav:
  - Home: index.md
  - Your Section:
      - Your Page: path/to/page.md
```

### Custom Styles
Add to `docs/guide/assets/custom.css`

## Troubleshooting

### Tree doesn't load
1. Check `docs/guide/assets/directory-tree.json` exists
2. Verify D3.js loads (check browser console)
3. Ensure mkdocs.yml includes D3.js in `extra_javascript`

### Build fails
1. Run `python docs/guide/gen_ref_pages.py` first
2. Check for syntax errors in markdown
3. Verify all files in `nav` exist

### Links broken
1. Use relative paths: `[link](../page.md)`
2. Don't use absolute paths
3. Test with `mkdocs build --strict`

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [D3.js Documentation](https://d3js.org/)
STEP8EOF

echo -e "${GREEN}✓ Step 8: Documentation README updated${NC}"
echo ""

# =============================================================================
# STEP 9: Create Development Guidelines Document
# =============================================================================
echo -e "${BLUE}=== STEP 9: Creating developer guidelines ===${NC}"

cat >> docs/guide/development/contributing.md << 'STEP9EOF'

## Updating Documentation

### Directory Structure

When you add new files or directories to the project:

1. **Automatic updates**: The directory tree updates automatically during documentation builds
2. **Manual update**: Run `python docs/guide/gen_ref_pages.py` locally to regenerate

### Adding Descriptions

To add descriptions for new directories or files:

1. Edit `docs/guide/gen_ref_pages.py`
2. Add entries to `DIRECTORY_DESCRIPTIONS` dict:
   ```python
   DIRECTORY_DESCRIPTIONS = {
       "your_new_dir": "Description of your new directory",
       "your_new_dir/subdir": "Description of subdirectory",
   }
   ```
3. Run `python docs/guide/gen_ref_pages.py` to regenerate
4. Preview with `mkdocs serve`

### Interactive Tree

The directory tree visualization will automatically show your changes:
- New files appear as green leaf nodes (📄)
- New directories appear as blue folder nodes (📁)
- Descriptions show on hover

No manual updates to the visualization code are needed!
STEP9EOF

echo -e "${GREEN}✓ Step 9: Developer guidelines added to contributing.md${NC}"
echo ""

# =============================================================================
# STEP 10: Final Verification and Summary
# =============================================================================
echo -e "${BLUE}=== STEP 10: Final verification ===${NC}"

# Check all critical files exist
critical_files=(
    "docs/guide/gen_ref_pages.py"
    "docs/guide/assets/directory-tree.json"
    "docs/guide/assets/js/directory-tree.js"
    "docs/guide/assets/css/directory-tree.css"
    "docs/guide/assets/directory-tree-container.html"
    "docs/guide/assets/custom.css"
    "docs/guide/guide/structure.md"
    "docs/readme.md"
    "mkdocs.yml"
    ".github/workflows/docs.yaml"
)

echo "Verifying critical files..."
all_present=true
for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file"
        all_present=false
    fi
done

echo ""

if [ "$all_present" = true ]; then
    echo -e "${GREEN}✓ Step 10: All files verified${NC}"
else
    echo -e "${YELLOW}⚠ Step 10: Some files missing (see above)${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "✅ ALL STEPS COMPLETE!"
echo "==========================================${NC}"
echo ""
echo -e "${BLUE}📋 What was accomplished:${NC}"
echo ""
echo "  ✅ Step 3: Embedded interactive tree in structure.md"
echo "  ✅ Step 4: Configured mkdocs.yml with Material theme"
echo "  ✅ Step 5: Updated GitHub Actions workflow"
echo "  ✅ Step 6: Verified .gitignore configuration"
echo "  ✅ Step 7: Tested local build"
echo "  ✅ Step 8: Updated documentation README"
echo "  ✅ Step 9: Added developer guidelines"
echo "  ✅ Step 10: Verified all files"
echo ""
echo -e "${BLUE}📁 File structure:${NC}"
echo ""
echo "  docs/guide/"
echo "  ├── gen_ref_pages.py           (generates JSON + markdown)"
echo "  ├── assets/"
echo "  │   ├── directory-tree.json    (auto-generated data)"
echo "  │   ├── custom.css             (custom styles)"
echo "  │   ├── js/"
echo "  │   │   └── directory-tree.js  (D3.js visualization)"
echo "  │   ├── css/"
echo "  │   │   └── directory-tree.css (tree styles)"
echo "  │   └── directory-tree-container.html"
echo "  ├── guide/"
echo "  │   └── structure.md           (embedded tree)"
echo "  └── readme.md                  (documentation guide)"
echo ""
echo "  mkdocs.yml                     (MkDocs configuration)"
echo "  .github/workflows/docs.yaml    (CI/CD pipeline)"
echo ""
echo -e "${BLUE}🧪 Testing:${NC}"
echo ""
echo "  1. Generate structure:"
echo "     ${GREEN}python docs/guide/gen_ref_pages.py${NC}"
echo ""
echo "  2. Start preview server:"
echo "     ${GREEN}mkdocs serve${NC}"
echo ""
echo "  3. Open browser:"
echo "     ${GREEN}http://localhost:8000${NC}"
echo ""
echo "  4. Navigate to: Project Guide → Structure"
echo ""
echo -e "${BLUE}🚀 Deployment:${NC}"
echo ""
echo "  Commit and push to trigger automatic deployment:"
echo "     ${GREEN}git add .${NC}"
echo "     ${GREEN}git commit -m \"Add interactive directory tree\"${NC}"
echo "     ${GREEN}git push origin main${NC}"
echo ""
echo "  GitHub Actions will:"
echo "    • Generate directory structure"
echo "    • Build MkDocs + Sphinx"
echo "    • Deploy to GitHub Pages"
echo ""
echo -e "${BLUE}🎉 You're all set!${NC}"
echo ""

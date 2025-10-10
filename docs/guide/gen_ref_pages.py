"""Generate the project structure reference pages and JSON data."""

from pathlib import Path
import os
import json

# Directory descriptions - shared between markdown and JSON
DIRECTORY_DESCRIPTIONS = {
    "core": "Foundation classes, base processors, configuration, and constants",
    "processors": "Specialized processors for each workflow stage",
    "processors/aoi": "Area of Interest (AOI) creation and management",
    "processors/spatial": "Spatial operations (clipping, alignment, reprojection)",
    "processors/extraction": "Data extraction from archives and downloads",
    "processors/integration": "Statistical enrichment and data integration",
    "processors/visualization": "Map and chart generation",
    "schemas": "Pydantic models for configuration validation",
    "utils": "Helper functions and common operations",
    "cli": "Command-line interface entry points",
    "cli/commands": "CLI command implementations",
    "visualization": "Visualization components",
    "visualization/raster": "Raster visualization processors",
    "visualization/vector": "Vector visualization processors",
    "visualization/reports": "Report generation utilities",
}

# File descriptions for specific important files
FILE_DESCRIPTIONS = {
    "__init__.py": "Package initialization",
    "__version__.py": "Version information",
    "base.py": "Abstract base classes for processors",
    "config.py": "Configuration management",
    "constants.py": "Project-wide constants",
    "exceptions.py": "Custom exception classes",
    "pipeline.py": "Processing pipeline orchestration",
    "config_models.py": "Pydantic configuration models",
    "main.py": "CLI entry point",
}


def build_tree_structure(root_path: Path, parent_path: str = "") -> dict:
    """
    Recursively build a hierarchical tree structure.
    
    Args:
        root_path: Path to the root directory
        parent_path: Relative path from src/geoworkflow (for descriptions)
        
    Returns:
        Dictionary representing the tree node
    """
    node = {
        "name": root_path.name,
        "type": "directory" if root_path.is_dir() else "file",
        "path": str(root_path.relative_to(root_path.parent.parent.parent)),
    }
    
    # Add description if available
    if parent_path in DIRECTORY_DESCRIPTIONS:
        node["description"] = DIRECTORY_DESCRIPTIONS[parent_path]
    elif node["name"] in FILE_DESCRIPTIONS:
        node["description"] = FILE_DESCRIPTIONS[node["name"]]
    
    # If it's a directory, add children
    if root_path.is_dir():
        children = []
        
        # Get all items, filter out hidden and cache
        items = sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for item in items:
            # Skip hidden files and __pycache__
            if item.name.startswith('.') or item.name == '__pycache__':
                continue
            
            # For files, only include .py files
            if item.is_file() and not item.name.endswith('.py'):
                continue
            
            # Build child path for description lookup
            child_rel_path = str(item.relative_to(root_path.parent.parent))
            child_rel_path = child_rel_path.replace('geoworkflow/', '').replace('geoworkflow', '')
            if child_rel_path.startswith('/'):
                child_rel_path = child_rel_path[1:]
            
            child_node = build_tree_structure(item, child_rel_path)
            children.append(child_node)
        
        if children:
            node["children"] = children
    
    return node


def generate_directory_json():
    """Generate JSON file with directory structure."""
    
    src_dir = Path("src/geoworkflow")
    output_file = Path("docs/guide/assets/directory-tree.json")
    
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return
    
    print(f"Scanning directory structure: {src_dir}")
    
    # Build the tree
    tree_data = build_tree_structure(src_dir, "")
    
    # Wrap in a root object with metadata
    output_data = {
        "name": "geoworkflow",
        "type": "directory",
        "description": "Root package for geospatial workflow processing",
        "children": tree_data.get("children", []),
        "metadata": {
            "generated_at": "auto-generated",
            "source_directory": str(src_dir),
        }
    }
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON with pretty formatting
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Generated JSON: {output_file}")


def generate_structure_page():
    """Generate directory structure documentation (markdown)."""
    
    src_dir = Path("src/geoworkflow")
    output_file = Path("docs/guide/guide/structure.md")
    
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return
    
    with open(output_file, "w") as f:
        f.write("# Project Structure\n\n")
        f.write("This page documents the organization of the GeoWorkflow codebase.\n\n")
        
        f.write("## Source Code Layout\n\n")
        f.write("```\n")
        f.write("src/geoworkflow/\n")
        
        # Walk through directory structure
        for root, dirs, files in os.walk(src_dir):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            level = root.replace(str(src_dir), "").count(os.sep)
            indent = "│   " * level
            folder_name = os.path.basename(root)
            
            if level > 0:
                f.write(f"{indent}├── {folder_name}/\n")
            
            # List files
            subindent = "│   " * (level + 1)
            for file in sorted(files):
                if not file.startswith('.') and file.endswith('.py'):
                    f.write(f"{subindent}├── {file}\n")
        
        f.write("```\n\n")
        
        # Add descriptions for main directories
        f.write("## Directory Descriptions\n\n")
        
        for path, desc in DIRECTORY_DESCRIPTIONS.items():
            # Only show top-level and one-level-deep descriptions in markdown
            if path.count('/') > 1:
                continue
                
            full_path = src_dir / path
            if full_path.exists():
                f.write(f"### `{path}/`\n\n")
                f.write(f"{desc}\n\n")
                
                # Check for README
                readme = full_path / "README.md"
                if readme.exists():
                    f.write(f"📄 *See [{path}/README.md](./{path.replace('/', '-')}-readme.md) for detailed information*\n\n")
    
    print(f"✓ Generated markdown: {output_file}")


# Run both generation functions
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Project Structure Documentation")
    print("=" * 60)
    
    generate_structure_page()
    generate_directory_json()
    
    print("\n" + "=" * 60)
    print("✓ All documentation files generated successfully!")
    print("=" * 60)

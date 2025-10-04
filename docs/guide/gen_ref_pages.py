"""Generate the project structure reference pages."""

from pathlib import Path
import os

def generate_structure_page():
    """Generate directory structure documentation."""
    
    src_dir = Path("src/geoworkflow")
    output_file = Path("docs/guide/guide/structure.md")
    
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
        
        descriptions = {
            "core": "**Core modules** - Foundation classes, base processors, configuration, and constants",
            "processors": "**Data processors** - Specialized processors for each workflow stage",
            "processors/aoi": "Area of Interest (AOI) creation and management",
            "processors/spatial": "Spatial operations (clipping, alignment, reprojection)",
            "processors/extraction": "Data extraction from archives and downloads",
            "processors/integration": "Statistical enrichment and data integration",
            "processors/visualization": "Map and chart generation",
            "schemas": "**Configuration schemas** - Pydantic models for validation",
            "utils": "**Utility modules** - Helper functions and common operations",
            "cli": "**Command-line interface** - Entry points for CLI commands",
        }
        
        for path, desc in descriptions.items():
            full_path = src_dir / path
            if full_path.exists():
                f.write(f"### `{path}/`\n\n")
                f.write(f"{desc}\n\n")
                
                # Check for README
                readme = full_path / "README.md"
                if readme.exists():
                    f.write(f"📄 *See [{path}/README.md](./{path.replace('/', '-')}-readme.md) for detailed information*\n\n")

# Run the generation
if __name__ == "__main__":
    generate_structure_page()
    print("✓ Generated structure documentation")
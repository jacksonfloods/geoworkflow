"""
Temporal raster utilities for aggregating time-series raster data.

This module provides utilities for computing temporal statistics (mean, median, sum, etc.)
from monthly or periodic raster files. It's designed to be Jupyter-friendly with progress
tracking and informative output.

Key features:
- Flexible month detection (filename patterns, metadata, manual mapping)
- Strict validation (ensures complete monthly coverage)
- Multiple statistics (mean, median, sum, min, max, std)
- Custom period support (seasonal, custom date ranges)
- Spatial consistency checking
- Progress tracking with tqdm

Example:
    >>> from geoworkflow.utils.temporal_raster_utils import compute_temporal_average
    >>> result = compute_temporal_average(
    ...     input_dir="data/monthly_tiffs/",
    ...     output_path="output/annual_mean.tif",
    ...     statistic="mean"
    ... )
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import re
import logging
from datetime import datetime
import warnings

try:
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import pandas as pd
    from tqdm.auto import tqdm
    HAS_REQUIRED_LIBS = True
except ImportError:
    HAS_REQUIRED_LIBS = False

logger = logging.getLogger(__name__)


# ==============================================================================
# Month Detection Functions
# ==============================================================================

def detect_month_from_filename(
    filepath: Path,
    patterns: Optional[List[str]] = None
) -> Optional[int]:
    """
    Detect month number (1-12) from filename using regex patterns.

    Args:
        filepath: Path to the TIFF file
        patterns: List of regex patterns to try. If None, uses common patterns.
                 Pattern should have a group capturing the month (1-12 or 01-12).

    Returns:
        Month number (1-12) if detected, None otherwise

    Examples:
        >>> detect_month_from_filename(Path("data_2022_03.tif"))
        3
        >>> detect_month_from_filename(Path("KEN_Nairobi_odiac_2022_12.tif"))
        12
    """
    if patterns is None:
        # Common patterns for month detection
        patterns = [
            r"_(\d{4})_(\d{2})(?:_|\.)(?:tif|TIF)$",  # YYYY_MM.tif
            r"_(\d{4})-(\d{2})(?:_|\.)(?:tif|TIF)$",  # YYYY-MM.tif
            r"_(\d{2})(?:_|\.)(?:tif|TIF)$",          # _MM.tif (end of filename)
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",  # Month names
        ]

    filename = filepath.name.lower()

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            # Handle numeric month
            if match.lastindex >= 2:
                # Pattern with YYYY_MM
                month_str = match.group(2)
            else:
                # Pattern with just MM
                month_str = match.group(1)

            # Handle month names
            if not month_str.isdigit():
                month_map = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month_str = month_map.get(month_str[:3])
                if month_str is None:
                    continue
            else:
                month_str = int(month_str)

            # Validate month range
            if 1 <= month_str <= 12:
                return month_str

    return None


def detect_month_from_metadata(
    filepath: Path,
    metadata_key: str = "GEOWORKFLOW_TEMPORAL_PERIOD"
) -> Optional[int]:
    """
    Detect month number from TIFF metadata tags.

    Args:
        filepath: Path to the TIFF file
        metadata_key: Metadata key to read (should contain YYYY-MM or similar)

    Returns:
        Month number (1-12) if detected, None otherwise
    """
    try:
        with rasterio.open(filepath) as src:
            metadata = src.tags()
            if metadata_key in metadata:
                period = metadata[metadata_key]
                # Try to parse YYYY-MM format
                match = re.search(r"(\d{4})-(\d{2})", period)
                if match:
                    month = int(match.group(2))
                    if 1 <= month <= 12:
                        return month
    except Exception as e:
        logger.debug(f"Failed to read metadata from {filepath}: {e}")

    return None


def auto_detect_months(
    filepaths: List[Path],
    filename_patterns: Optional[List[str]] = None,
    metadata_key: str = "GEOWORKFLOW_TEMPORAL_PERIOD",
    skip_undetected: bool = True
) -> Dict[int, Path]:
    """
    Automatically detect month assignments for a list of files.

    Tries filename pattern first, then metadata if filename fails.

    Args:
        filepaths: List of TIFF file paths
        filename_patterns: Regex patterns for filename detection
        metadata_key: Metadata key for metadata detection
        skip_undetected: If True, skip files where month can't be detected (with warning).
                        If False, raise error for undetected files.

    Returns:
        Dictionary mapping month number (1-12) to filepath

    Raises:
        ValueError: If months cannot be detected (when skip_undetected=False) or there are duplicates
    """
    month_to_file = {}
    undetected = []

    for filepath in filepaths:
        # Try filename detection first
        month = detect_month_from_filename(filepath, filename_patterns)

        # Fall back to metadata
        if month is None:
            month = detect_month_from_metadata(filepath, metadata_key)

        if month is None:
            undetected.append(filepath.name)
        elif month in month_to_file:
            raise ValueError(
                f"Duplicate month {month} detected:\n"
                f"  File 1: {month_to_file[month].name}\n"
                f"  File 2: {filepath.name}"
            )
        else:
            month_to_file[month] = filepath

    if undetected:
        if skip_undetected:
            # Just warn, don't error
            logger.info(
                f"Skipped {len(undetected)} file(s) with undetectable months: " +
                ", ".join(undetected[:3]) +
                (f" and {len(undetected)-3} more" if len(undetected) > 3 else "")
            )
        else:
            raise ValueError(
                f"Could not detect months for {len(undetected)} file(s):\n" +
                "\n".join(f"  - {name}" for name in undetected[:5]) +
                (f"\n  ... and {len(undetected)-5} more" if len(undetected) > 5 else "")
            )

    return month_to_file


# ==============================================================================
# Validation Functions
# ==============================================================================

def validate_monthly_completeness(
    detected_months: Dict[int, Path],
    required_months: Optional[List[int]] = None
) -> Tuple[bool, List[int]]:
    """
    Validate that all required months are present.

    Args:
        detected_months: Dictionary mapping month -> filepath
        required_months: List of required month numbers (1-12).
                        If None, requires all 12 months.

    Returns:
        Tuple of (is_complete, missing_months)
    """
    if required_months is None:
        required_months = list(range(1, 13))

    missing = [m for m in required_months if m not in detected_months]
    return len(missing) == 0, missing


def check_spatial_consistency(
    filepaths: List[Path]
) -> Tuple[bool, Optional[str]]:
    """
    Check that all TIFFs have consistent spatial properties.

    Validates:
    - Same CRS
    - Same dimensions (width, height)
    - Same transform (pixel size and origin)

    Args:
        filepaths: List of TIFF files to check

    Returns:
        Tuple of (is_consistent, error_message)
    """
    if not filepaths:
        return True, None

    # Read properties from first file as reference
    try:
        with rasterio.open(filepaths[0]) as ref:
            ref_crs = ref.crs
            ref_shape = (ref.height, ref.width)
            ref_transform = ref.transform
            ref_nodata = ref.nodata
    except Exception as e:
        return False, f"Failed to read reference file {filepaths[0].name}: {e}"

    # Check all other files
    for filepath in filepaths[1:]:
        try:
            with rasterio.open(filepath) as src:
                if src.crs != ref_crs:
                    return False, (
                        f"CRS mismatch:\n"
                        f"  {filepaths[0].name}: {ref_crs}\n"
                        f"  {filepath.name}: {src.crs}"
                    )

                if (src.height, src.width) != ref_shape:
                    return False, (
                        f"Dimension mismatch:\n"
                        f"  {filepaths[0].name}: {ref_shape}\n"
                        f"  {filepath.name}: ({src.height}, {src.width})"
                    )

                if src.transform != ref_transform:
                    return False, (
                        f"Transform mismatch:\n"
                        f"  {filepaths[0].name}: {ref_transform}\n"
                        f"  {filepath.name}: {src.transform}"
                    )
        except Exception as e:
            return False, f"Failed to read {filepath.name}: {e}"

    return True, None


# ==============================================================================
# Statistics Functions
# ==============================================================================

def compute_statistic(
    arrays: np.ndarray,
    statistic: str,
    nodata: Optional[float] = None
) -> np.ndarray:
    """
    Compute temporal statistic across stacked arrays.

    Args:
        arrays: 3D array (time, height, width)
        statistic: One of: mean, median, sum, min, max, std
        nodata: Nodata value to mask

    Returns:
        2D array with computed statistic
    """
    # Mask nodata if present
    if nodata is not None:
        masked = np.ma.masked_equal(arrays, nodata)
    else:
        masked = np.ma.masked_invalid(arrays)

    # Compute statistic along time axis (axis=0)
    if statistic == "mean":
        result = masked.mean(axis=0)
    elif statistic == "median":
        result = np.ma.median(masked, axis=0)
    elif statistic == "sum":
        result = masked.sum(axis=0)
    elif statistic == "min":
        result = masked.min(axis=0)
    elif statistic == "max":
        result = masked.max(axis=0)
    elif statistic == "std":
        result = masked.std(axis=0)
    else:
        raise ValueError(
            f"Unknown statistic '{statistic}'. "
            f"Choose from: mean, median, sum, min, max, std"
        )

    # Fill masked values with nodata
    if nodata is not None:
        result = result.filled(nodata)
    else:
        result = result.filled(np.nan)

    return result


# ==============================================================================
# Main Function
# ==============================================================================

def compute_temporal_average(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    statistic: str = "mean",
    month_detection: str = "auto",
    filename_pattern: Optional[Union[str, List[str]]] = None,
    metadata_key: str = "GEOWORKFLOW_TEMPORAL_PERIOD",
    file_month_mapping: Optional[Dict[str, int]] = None,
    period_months: Optional[List[int]] = None,
    require_complete: bool = True,
    compression: str = "lzw",
    overwrite: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute temporal average from monthly TIFF files.

    This function is designed to be Jupyter-friendly with progress bars
    and informative output. It validates spatial consistency and month
    completeness before processing.

    Args:
        input_dir: Directory containing monthly TIFF files
        output_path: Path for output TIFF file
        statistic: Temporal statistic to compute
                  Options: mean, median, sum, min, max, std
        month_detection: How to detect months from files
                        Options: auto, filename, metadata, manual
        filename_pattern: Regex pattern(s) for filename detection
        metadata_key: TIFF metadata tag key for month info
        file_month_mapping: Manual mapping of filename -> month number
                           e.g., {"file1.tif": 1, "file2.tif": 2, ...}
        period_months: List of month numbers to include (1-12)
                      None = all 12 months (default)
                      [12,1,2] = Dec-Jan-Feb (winter)
        require_complete: If True, error if not all months present
        compression: Output TIFF compression (lzw, deflate, none)
        overwrite: Overwrite existing output file
        verbose: Print progress and summary information

    Returns:
        Dictionary with processing results:
        - success: bool
        - output_file: Path
        - detected_months: Dict[int, Path]
        - statistic: str
        - input_count: int
        - message: str

    Raises:
        ValueError: If validation fails or inputs are invalid
        FileNotFoundError: If input directory doesn't exist
        FileExistsError: If output exists and overwrite=False

    Example:
        >>> # Automatic detection, annual mean
        >>> result = compute_temporal_average(
        ...     input_dir="data/monthly/",
        ...     output_path="output/annual_mean.tif"
        ... )
        >>>
        >>> # Winter (DJF) median
        >>> result = compute_temporal_average(
        ...     input_dir="data/monthly/",
        ...     output_path="output/winter_median.tif",
        ...     statistic="median",
        ...     period_months=[12, 1, 2]
        ... )
    """
    if not HAS_REQUIRED_LIBS:
        raise ImportError(
            "Required libraries not available. Install with:\n"
            "pip install numpy rasterio pandas tqdm"
        )

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Validate inputs
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}\n"
                f"Set overwrite=True to replace it."
            )
        else:
            # Explicitly delete the file to avoid issues with corrupted files
            output_path.unlink()

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all TIFF files
    tiff_files = sorted([
        f for f in input_dir.glob("*.tif")
        if f.is_file() and not f.name.startswith('.')
    ])

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    if verbose:
        print(f"Found {len(tiff_files)} TIFF file(s) in {input_dir}")
        print()

    # Detect months
    if month_detection == "manual":
        if file_month_mapping is None:
            raise ValueError(
                "file_month_mapping is required when month_detection='manual'"
            )
        # Convert filename strings to Paths
        month_to_file = {}
        for filename, month in file_month_mapping.items():
            filepath = input_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            month_to_file[month] = filepath

    elif month_detection == "filename":
        patterns = [filename_pattern] if isinstance(filename_pattern, str) else filename_pattern
        month_to_file = {}
        for filepath in tiff_files:
            month = detect_month_from_filename(filepath, patterns)
            if month:
                month_to_file[month] = filepath

    elif month_detection == "metadata":
        month_to_file = {}
        for filepath in tiff_files:
            month = detect_month_from_metadata(filepath, metadata_key)
            if month:
                month_to_file[month] = filepath

    elif month_detection == "auto":
        patterns = [filename_pattern] if isinstance(filename_pattern, str) else filename_pattern
        month_to_file = auto_detect_months(tiff_files, patterns, metadata_key)
    else:
        raise ValueError(
            f"Invalid month_detection '{month_detection}'. "
            f"Choose from: auto, filename, metadata, manual"
        )

    # Display detected months
    if verbose and month_to_file:
        df = pd.DataFrame([
            {"Month": m, "File": p.name}
            for m, p in sorted(month_to_file.items())
        ])
        print("Detected months:")
        print(df.to_string(index=False))
        print()

    # Validate completeness
    required_months = period_months if period_months else list(range(1, 13))
    is_complete, missing = validate_monthly_completeness(month_to_file, required_months)

    if not is_complete:
        if require_complete:
            raise ValueError(
                f"Incomplete monthly data. Missing months: {missing}\n"
                f"Set require_complete=False to process anyway."
            )
        else:
            warnings.warn(
                f"Processing with incomplete data. Missing months: {missing}"
            )

    # Filter to requested period
    if period_months:
        month_to_file = {
            m: p for m, p in month_to_file.items()
            if m in period_months
        }

    if not month_to_file:
        raise ValueError("No files match the requested period")

    # Check spatial consistency
    file_list = [month_to_file[m] for m in sorted(month_to_file.keys())]
    is_consistent, error_msg = check_spatial_consistency(file_list)

    if not is_consistent:
        raise ValueError(f"Spatial inconsistency detected:\n{error_msg}")

    if verbose:
        print(f"✓ Spatial consistency validated")
        print(f"✓ Processing {len(month_to_file)} months with statistic: {statistic}")
        print()

    # Read all monthly data
    arrays = []
    month_list = sorted(month_to_file.keys())

    with rasterio.open(month_to_file[month_list[0]]) as ref:
        # Create clean profile with only essential fields
        profile = {
            'driver': 'GTiff',
            'dtype': ref.dtypes[0],
            'width': ref.width,
            'height': ref.height,
            'count': 1,
            'crs': ref.crs,
            'transform': ref.transform,
            'nodata': ref.nodata
        }
        nodata = ref.nodata

    for month in tqdm(month_list, desc="Reading monthly TIFFs", disable=not verbose):
        filepath = month_to_file[month]
        with rasterio.open(filepath) as src:
            data = src.read(1)
            arrays.append(data)

    # Stack and compute statistic
    stacked = np.stack(arrays, axis=0)
    if verbose:
        print(f"Computing {statistic}...")

    result_array = compute_statistic(stacked, statistic, nodata)

    # Write output
    # Only use tiling if dimensions are suitable (multiples of 16)
    use_tiling = (profile['height'] % 16 == 0 and profile['width'] % 16 == 0)

    profile.update(
        compress=compression,
        tiled=use_tiling
    )

    if use_tiling:
        # Set reasonable tile size (256x256 is standard)
        profile.update(blockxsize=256, blockysize=256)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(result_array, 1)

        # Add metadata
        dst.update_tags(
            GEOWORKFLOW_TEMPORAL_STATISTIC=statistic,
            GEOWORKFLOW_TEMPORAL_MONTHS=",".join(map(str, sorted(month_list))),
            GEOWORKFLOW_PROCESSED_DATE=datetime.now().isoformat(),
            GEOWORKFLOW_INPUT_COUNT=len(month_list)
        )

    if verbose:
        print(f"\n✓ Success! Output written to: {output_path}")
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")

    return {
        "success": True,
        "output_file": output_path,
        "detected_months": month_to_file,
        "statistic": statistic,
        "input_count": len(month_list),
        "message": f"Successfully computed {statistic} from {len(month_list)} months"
    }

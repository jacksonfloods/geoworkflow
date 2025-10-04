"""
Google Cloud Storage utilities for Open Buildings data access.
"""
from typing import Optional, Union, List, Any
from pathlib import Path
import gcsfs
import logging
import pandas as pd


class GCSClient:
    """
    Wrapper for GCS access with multiple auth methods.
    
    Supports:
    - Anonymous access (for public datasets like Open Buildings)
    - Service account authentication
    - Default credentials
    """
    
    def __init__(
        self,
        service_account_key: Optional[Path] = None,
        use_anonymous: bool = True
    ):
        """
        Initialize GCS client.
        
        Args:
            service_account_key: Path to service account JSON key file
            use_anonymous: Use anonymous access (for public data)
            
        Raises:
            ImportError: If gcsfs is not installed
            ValueError: If authentication fails
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize gcsfs filesystem
        try:
            if use_anonymous:
                self.fs = gcsfs.GCSFileSystem(token='anon')
                self.logger.info("Initialized GCS client with anonymous access")
            elif service_account_key:
                self.fs = gcsfs.GCSFileSystem(token=str(service_account_key))
                self.logger.info(f"Initialized GCS client with service account: {service_account_key}")
            else:
                self.fs = gcsfs.GCSFileSystem()
                self.logger.info("Initialized GCS client with default credentials")
                
        except Exception as e:
            raise ValueError(f"Failed to initialize GCS client: {e}")
    
    def file_exists(self, path: str) -> bool:
        """
        Check if GCS file exists.
        
        Args:
            path: GCS path (e.g., 'gs://bucket/path/file.csv.gz')
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            # Remove gs:// prefix if present
            path = path.replace('gs://', '')
            return self.fs.exists(path)
        except Exception as e:
            self.logger.warning(f"Error checking if file exists {path}: {e}")
            return False
    
    def THIS SOFTWARE NEEDS AN OPEN SOURCE LICENSE
        self, 
        path: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Read compressed CSV from GCS.
        
        Args:
            path: GCS path to CSV.gz file
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            DataFrame with CSV contents
            
        Example:
            >>> client = GCSClient(use_anonymous=True)
            >>> df = client.THIS SOFTWARE NEEDS AN OPEN SOURCE LICENSE'gs://bucket/data.csv.gz')
        """
        # Remove gs:// prefix if present
        path = path.replace('gs://', '')
        
        # Default column names for Open Buildings v3 polygons (no header files)
        default_names = ['latitude', 'longitude', 'area_in_meters', 'confidence', 'geometry', 'full_plus_code']
        
        # Allow override via kwargs
        if 'names' not in kwargs:
            kwargs['names'] = default_names
        if 'header' not in kwargs:
            kwargs['header'] = None  # No header row
        
        with self.fs.open(path, 'rb') as f:
            return pd.read_csv(f, compression='gzip', **kwargs)
    
    def list_files(self, prefix: str, suffix: str = '') -> List[str]:
        """
        List files with given prefix.
        
        Args:
            prefix: GCS path prefix (e.g., 'gs://bucket/path/')
            suffix: Optional file suffix filter (e.g., '.csv.gz')
            
        Returns:
            List of GCS paths matching prefix and suffix
            
        Example:
            >>> client = GCSClient(use_anonymous=True)
            >>> files = client.list_files('gs://open-buildings-data/v3/', suffix='.csv.gz')
        """
        # Remove gs:// prefix if present
        prefix = prefix.replace('gs://', '')
        
        try:
            files = self.fs.ls(prefix)
            
            if suffix:
                files = [f for f in files if f.endswith(suffix)]
            
            # Add gs:// prefix back
            return [f'gs://{f}' for f in files]
            
        except Exception as e:
            self.logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []
    
    def download_file(self, gcs_path: str, local_path: Path) -> bool:
        """
        Download file from GCS to local path.
        
        Args:
            gcs_path: GCS path (e.g., 'gs://bucket/file.csv.gz')
            local_path: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        # Remove gs:// prefix if present
        gcs_path = gcs_path.replace('gs://', '')
        
        try:
            self.fs.get(gcs_path, str(local_path))
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {gcs_path}: {e}")
            return False
    
    def get_file_size(self, path: str) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            path: GCS path
            
        Returns:
            File size in bytes, or None if error
        """
        # Remove gs:// prefix if present
        path = path.replace('gs://', '')
        
        try:
            info = self.fs.info(path)
            return info.get('size')
        except Exception as e:
            self.logger.warning(f"Error getting file size for {path}: {e}")
            return None

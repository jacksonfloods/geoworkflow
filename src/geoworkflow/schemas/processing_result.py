"""Result classes for batch processing operations."""

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


@dataclass
class BatchProcessResult:
    """Results from batch processing multiple agglomerations."""
    success: bool
    total_count: int
    succeeded_count: int
    failed_count: int
    succeeded: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    output_files: List[Path] = field(default_factory=list)
    
    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            "=" * 60,
            "BATCH PROCESSING RESULT",
            "=" * 60,
            f"Status: {status}",
            f"Total agglomerations: {self.total_count}",
            f"Succeeded: {self.succeeded_count}",
            f"Failed: {self.failed_count}",
            f"Output files generated: {len(self.output_files)}",
        ]
        
        if self.succeeded and len(self.succeeded) <= 10:
            lines.append(f"\nSucceeded: {', '.join(self.succeeded)}")
        elif self.succeeded:
            lines.append(f"\nSucceeded: {', '.join(self.succeeded[:10])}... (+{len(self.succeeded)-10} more)")
        
        if self.failed and len(self.failed) <= 5:
            lines.append("\nFailed:")
            for name, error in self.failed.items():
                lines.append(f"  - {name}: {error}")
        elif self.failed:
            lines.append(f"\nFailed: {len(self.failed)} agglomerations (see logs for details)")
        
        lines.append("=" * 60)
        return "\n".join(lines)

"""
Application settings and configuration.
"""

from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Settings:
    """Application settings."""

    # Paths
    default_output_folder: Optional[Path] = None
    last_experiment_folder: Optional[Path] = None

    # Channel defaults
    default_nuclear_suffix: str = "C0"
    default_cyto_suffix: str = "C1"
    default_marker_suffixes: List[str] = field(default_factory=lambda: ["C2"])
    default_marker_names: List[str] = field(default_factory=lambda: ["Marker1"])

    # Segmentation defaults
    default_model: str = "cpsam"
    default_diameter: float = 30.0
    default_flow_threshold: float = 0.4
    default_min_size: int = 15

    # Performance
    use_gpu: bool = True
    batch_size: int = 4
    n_workers: int = 4
    use_vectorized_ctcf: bool = True

    # UI
    theme: str = "dark"
    show_advanced_options: bool = False

    def save(self, path: Path):
        """Save settings to JSON file."""
        data = {
            'default_output_folder': str(self.default_output_folder) if self.default_output_folder else None,
            'last_experiment_folder': str(self.last_experiment_folder) if self.last_experiment_folder else None,
            'default_nuclear_suffix': self.default_nuclear_suffix,
            'default_cyto_suffix': self.default_cyto_suffix,
            'default_marker_suffixes': self.default_marker_suffixes,
            'default_marker_names': self.default_marker_names,
            'default_model': self.default_model,
            'default_diameter': self.default_diameter,
            'default_flow_threshold': self.default_flow_threshold,
            'default_min_size': self.default_min_size,
            'use_gpu': self.use_gpu,
            'batch_size': self.batch_size,
            'n_workers': self.n_workers,
            'use_vectorized_ctcf': self.use_vectorized_ctcf,
            'theme': self.theme,
            'show_advanced_options': self.show_advanced_options,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Settings':
        """Load settings from JSON file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls(
            default_output_folder=Path(data['default_output_folder']) if data.get('default_output_folder') else None,
            last_experiment_folder=Path(data['last_experiment_folder']) if data.get('last_experiment_folder') else None,
            default_nuclear_suffix=data.get('default_nuclear_suffix', "C0"),
            default_cyto_suffix=data.get('default_cyto_suffix', "C1"),
            default_marker_suffixes=data.get('default_marker_suffixes', ["C2"]),
            default_marker_names=data.get('default_marker_names', ["Marker1"]),
            default_model=data.get('default_model', "cpsam"),
            default_diameter=data.get('default_diameter', 30.0),
            default_flow_threshold=data.get('default_flow_threshold', 0.4),
            default_min_size=data.get('default_min_size', 15),
            use_gpu=data.get('use_gpu', True),
            batch_size=data.get('batch_size', 4),
            n_workers=data.get('n_workers', 4),
            use_vectorized_ctcf=data.get('use_vectorized_ctcf', True),
            theme=data.get('theme', "dark"),
            show_advanced_options=data.get('show_advanced_options', False),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        config_path = Path.home() / ".cellquant_enterprise" / "settings.json"
        _settings = Settings.load(config_path)
    return _settings


def save_settings(settings: Settings):
    """Save settings to the default config path."""
    global _settings
    config_path = Path.home() / ".cellquant_enterprise" / "settings.json"
    settings.save(config_path)
    _settings = settings

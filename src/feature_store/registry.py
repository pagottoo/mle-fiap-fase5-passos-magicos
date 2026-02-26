"""
Feature Registry - feature definitions and metadata.
"""
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger()


@dataclass
class FeatureDefinition:
    """Feature definition."""
    name: str
    dtype: str  # "numeric", "categorical", "boolean"
    description: str
    source: str  # Source column in raw data
    transformation: Optional[str] = None  # Applied transformation
    version: str = "1.0.0"
    created_at: str = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []


@dataclass
class FeatureGroup:
    """Group of related features."""
    name: str
    description: str
    features: List[str]
    entity: str  # Main entity (e.g. "aluno")
    version: str = "1.0.0"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class FeatureRegistry:
    """
    Centralized registry for feature definitions.

    Stores metadata for all available features,
    their transformations, and group relationships.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize registry.

        Args:
            registry_path: JSON registry file path.
        """
        self.registry_path = registry_path or Path(__file__).parent.parent.parent / "feature_store" / "registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._features: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        
        self._load_registry()
        self._register_default_features()
    
    def _load_registry(self) -> None:
        """Load registry file from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                
                for name, feat_data in data.get("features", {}).items():
                    self._features[name] = FeatureDefinition(**feat_data)
                
                for name, group_data in data.get("groups", {}).items():
                    self._groups[name] = FeatureGroup(**group_data)
                
                logger.info("registry_loaded", features=len(self._features), groups=len(self._groups))
            except Exception as e:
                logger.warning("registry_load_failed", error=str(e))
    
    def _save_registry(self) -> None:
        """Persist registry file to disk."""
        data = {
            "features": {name: asdict(feat) for name, feat in self._features.items()},
            "groups": {name: asdict(group) for name, group in self._groups.items()},
            "updated_at": datetime.now().isoformat()
        }
        
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info("registry_saved", path=str(self.registry_path))
    
    def _register_default_features(self) -> None:
        """Register default features for this project."""
        # Numeric features - educational indexes
        numeric_features = [
            FeatureDefinition(
                name="inde",
                dtype="numeric",
                description="Educational Development Index",
                source="INDE",
                transformation="standard_scaling",
                tags=["index", "development"]
            ),
            FeatureDefinition(
                name="iaa",
                dtype="numeric",
                description="Self-Assessment Index",
                source="IAA",
                transformation="standard_scaling",
                tags=["index", "self_assessment"]
            ),
            FeatureDefinition(
                name="ieg",
                dtype="numeric",
                description="Engagement Index",
                source="IEG",
                transformation="standard_scaling",
                tags=["index", "engagement"]
            ),
            FeatureDefinition(
                name="ips",
                dtype="numeric",
                description="Psychosocial Index",
                source="IPS",
                transformation="standard_scaling",
                tags=["index", "psychosocial"]
            ),
            FeatureDefinition(
                name="ida",
                dtype="numeric",
                description="Academic Performance Index",
                source="IDA",
                transformation="standard_scaling",
                tags=["index", "academic"]
            ),
            FeatureDefinition(
                name="ipp",
                dtype="numeric",
                description="Participation Index",
                source="IPP",
                transformation="standard_scaling",
                tags=["index", "participation"]
            ),
            FeatureDefinition(
                name="ipv",
                dtype="numeric",
                description="Turning Point Propensity Index",
                source="IPV",
                transformation="standard_scaling",
                tags=["index", "turning_point", "target_related"]
            ),
            FeatureDefinition(
                name="ian",
                dtype="numeric",
                description="Level Adequacy Index",
                source="IAN",
                transformation="standard_scaling",
                tags=["index", "level"]
            ),
            FeatureDefinition(
                name="idade_aluno",
                dtype="numeric",
                description="Student age in years",
                source="IDADE_ALUNO",
                transformation="standard_scaling",
                tags=["demographic"]
            ),
            FeatureDefinition(
                name="anos_pm",
                dtype="numeric",
                description="Years participating in Passos Magicos",
                source="ANOS_PM",
                transformation="standard_scaling",
                tags=["history", "engagement"]
            ),
        ]
        
        # Categorical features
        categorical_features = [
            FeatureDefinition(
                name="instituicao_ensino",
                dtype="categorical",
                description="Student school institution type",
                source="INSTITUICAO_ENSINO_ALUNO",
                transformation="label_encoding",
                tags=["demographic", "school"]
            ),
            FeatureDefinition(
                name="fase",
                dtype="categorical",
                description="Passos Magicos program stage",
                source="FASE",
                transformation="label_encoding",
                tags=["program", "level"]
            ),
            FeatureDefinition(
                name="pedra",
                dtype="categorical",
                description="Student cluster (Quartzo, Ametista, Agata, Topazio)",
                source="PEDRA",
                transformation="label_encoding",
                tags=["classification", "performance"]
            ),
            FeatureDefinition(
                name="bolsista",
                dtype="categorical",
                description="Indicates scholarship status",
                source="BOLSISTA",
                transformation="label_encoding",
                tags=["demographic", "scholarship"]
            ),
        ]
        
        # Register all features
        for feat in numeric_features + categorical_features:
            if feat.name not in self._features:
                self._features[feat.name] = feat
        
        # Create feature groups
        if "indices_avaliacao" not in self._groups:
            self._groups["indices_avaliacao"] = FeatureGroup(
                name="indices_avaliacao",
                description="Educational evaluation indexes",
                features=["inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian"],
                entity="aluno"
            )
        
        if "demografico" not in self._groups:
            self._groups["demografico"] = FeatureGroup(
                name="demografico",
                description="Student demographic characteristics",
                features=["idade_aluno", "anos_pm", "instituicao_ensino", "bolsista"],
                entity="aluno"
            )
        
        if "ponto_virada_features" not in self._groups:
            self._groups["ponto_virada_features"] = FeatureGroup(
                name="ponto_virada_features",
                description="Features for Turning Point prediction",
                features=[
                    "inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian",
                    "idade_aluno", "anos_pm", "instituicao_ensino", "fase", "pedra", "bolsista"
                ],
                entity="aluno"
            )
        
        self._save_registry()
    
    def register_feature(self, feature: FeatureDefinition) -> None:
        """
        Register a new feature.

        Args:
            feature: Feature definition.
        """
        self._features[feature.name] = feature
        self._save_registry()
        logger.info("feature_registered", name=feature.name)
    
    def register_group(self, group: FeatureGroup) -> None:
        """
        Register a new feature group.

        Args:
            group: Group definition.
        """
        self._groups[group.name] = group
        self._save_registry()
        logger.info("group_registered", name=group.name)
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self._features.get(name)
    
    def get_group(self, name: str) -> Optional[FeatureGroup]:
        """Get feature group by name."""
        return self._groups.get(name)
    
    def get_features_by_group(self, group_name: str) -> List[FeatureDefinition]:
        """Get all features from a group."""
        group = self.get_group(group_name)
        if not group:
            return []
        
        return [self._features[name] for name in group.features if name in self._features]
    
    def get_features_by_tag(self, tag: str) -> List[FeatureDefinition]:
        """Get features by tag (supports PT/EN aliases)."""
        alias_map = {
            "indice": "index",
            "indices": "index",
            "index": "index",
        }

        def _normalize(value: str) -> str:
            normalized = unicodedata.normalize("NFKD", value or "")
            normalized = normalized.encode("ascii", "ignore").decode("ascii")
            return normalized.strip().lower()

        target_raw = _normalize(tag)
        if not target_raw:
            return []

        target_canonical = alias_map.get(target_raw, target_raw)

        matched: List[FeatureDefinition] = []
        for feat in self._features.values():
            for feat_tag in feat.tags:
                feat_tag_raw = _normalize(feat_tag)
                feat_tag_canonical = alias_map.get(feat_tag_raw, feat_tag_raw)
                if feat_tag_raw == target_raw or feat_tag_canonical == target_canonical:
                    matched.append(feat)
                    break

        return matched
    
    def list_features(self) -> List[str]:
        """List all feature names."""
        return list(self._features.keys())
    
    def list_groups(self) -> List[str]:
        """List all group names."""
        return list(self._groups.keys())
    
    def get_source_columns(self, feature_names: List[str]) -> Dict[str, str]:
        """
        Return mapping from feature names to source columns.

        Args:
            feature_names: Feature names.

        Returns:
            Dictionary `{feature_name: source_column}`.
        """
        mapping = {}
        for name in feature_names:
            feat = self.get_feature(name)
            if feat:
                mapping[name] = feat.source
        return mapping

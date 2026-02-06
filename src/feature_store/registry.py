"""
Feature Registry - Registro e definição de features
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger()


@dataclass
class FeatureDefinition:
    """Definição de uma feature."""
    name: str
    dtype: str  # "numeric", "categorical", "boolean"
    description: str
    source: str  # Coluna de origem nos dados
    transformation: Optional[str] = None  # Transformação aplicada
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
    """Grupo de features relacionadas."""
    name: str
    description: str
    features: List[str]
    entity: str  # Entidade principal (ex: "aluno")
    version: str = "1.0.0"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class FeatureRegistry:
    """
    Registro centralizado de definições de features.
    
    Mantém metadados sobre todas as features disponíveis,
    suas transformações e relacionamentos.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Inicializa o registro.
        
        Args:
            registry_path: Caminho para o arquivo de registro JSON
        """
        self.registry_path = registry_path or Path(__file__).parent.parent.parent / "feature_store" / "registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._features: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        
        self._load_registry()
        self._register_default_features()
    
    def _load_registry(self) -> None:
        """Carrega registro do arquivo."""
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
        """Salva registro no arquivo."""
        data = {
            "features": {name: asdict(feat) for name, feat in self._features.items()},
            "groups": {name: asdict(group) for name, group in self._groups.items()},
            "updated_at": datetime.now().isoformat()
        }
        
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info("registry_saved", path=str(self.registry_path))
    
    def _register_default_features(self) -> None:
        """Registra features padrão do projeto Passos Mágicos."""
        # Features numéricas - Índices de avaliação
        numeric_features = [
            FeatureDefinition(
                name="inde",
                dtype="numeric",
                description="Índice de Desenvolvimento Educacional",
                source="INDE",
                transformation="standard_scaling",
                tags=["indice", "desenvolvimento"]
            ),
            FeatureDefinition(
                name="iaa",
                dtype="numeric",
                description="Índice de Autoavaliação",
                source="IAA",
                transformation="standard_scaling",
                tags=["indice", "autoavaliacao"]
            ),
            FeatureDefinition(
                name="ieg",
                dtype="numeric",
                description="Índice de Engajamento",
                source="IEG",
                transformation="standard_scaling",
                tags=["indice", "engajamento"]
            ),
            FeatureDefinition(
                name="ips",
                dtype="numeric",
                description="Índice Psicossocial",
                source="IPS",
                transformation="standard_scaling",
                tags=["indice", "psicossocial"]
            ),
            FeatureDefinition(
                name="ida",
                dtype="numeric",
                description="Índice de Desempenho Acadêmico",
                source="IDA",
                transformation="standard_scaling",
                tags=["indice", "academico"]
            ),
            FeatureDefinition(
                name="ipp",
                dtype="numeric",
                description="Índice de Participação",
                source="IPP",
                transformation="standard_scaling",
                tags=["indice", "participacao"]
            ),
            FeatureDefinition(
                name="ipv",
                dtype="numeric",
                description="Índice de Propensão à Virada",
                source="IPV",
                transformation="standard_scaling",
                tags=["indice", "virada", "target_related"]
            ),
            FeatureDefinition(
                name="ian",
                dtype="numeric",
                description="Índice de Adequação ao Nível",
                source="IAN",
                transformation="standard_scaling",
                tags=["indice", "nivel"]
            ),
            FeatureDefinition(
                name="idade_aluno",
                dtype="numeric",
                description="Idade do aluno em anos",
                source="IDADE_ALUNO",
                transformation="standard_scaling",
                tags=["demografico"]
            ),
            FeatureDefinition(
                name="anos_pm",
                dtype="numeric",
                description="Anos de participação na Passos Mágicos",
                source="ANOS_PM",
                transformation="standard_scaling",
                tags=["historico", "engajamento"]
            ),
        ]
        
        # Features categóricas
        categorical_features = [
            FeatureDefinition(
                name="instituicao_ensino",
                dtype="categorical",
                description="Tipo de instituição de ensino do aluno",
                source="INSTITUICAO_ENSINO_ALUNO",
                transformation="label_encoding",
                tags=["demografico", "escola"]
            ),
            FeatureDefinition(
                name="fase",
                dtype="categorical",
                description="Fase do programa Passos Mágicos",
                source="FASE",
                transformation="label_encoding",
                tags=["programa", "nivel"]
            ),
            FeatureDefinition(
                name="pedra",
                dtype="categorical",
                description="Classificação do aluno (Quartzo, Ametista, Ágata, Topázio)",
                source="PEDRA",
                transformation="label_encoding",
                tags=["classificacao", "performance"]
            ),
            FeatureDefinition(
                name="bolsista",
                dtype="categorical",
                description="Indica se o aluno é bolsista",
                source="BOLSISTA",
                transformation="label_encoding",
                tags=["demografico", "bolsa"]
            ),
        ]
        
        # Registrar todas as features
        for feat in numeric_features + categorical_features:
            if feat.name not in self._features:
                self._features[feat.name] = feat
        
        # Criar grupos de features
        if "indices_avaliacao" not in self._groups:
            self._groups["indices_avaliacao"] = FeatureGroup(
                name="indices_avaliacao",
                description="Índices de avaliação educacional",
                features=["inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian"],
                entity="aluno"
            )
        
        if "demografico" not in self._groups:
            self._groups["demografico"] = FeatureGroup(
                name="demografico",
                description="Características demográficas do aluno",
                features=["idade_aluno", "anos_pm", "instituicao_ensino", "bolsista"],
                entity="aluno"
            )
        
        if "ponto_virada_features" not in self._groups:
            self._groups["ponto_virada_features"] = FeatureGroup(
                name="ponto_virada_features",
                description="Features para predição de Ponto de Virada",
                features=[
                    "inde", "iaa", "ieg", "ips", "ida", "ipp", "ipv", "ian",
                    "idade_aluno", "anos_pm", "instituicao_ensino", "fase", "pedra", "bolsista"
                ],
                entity="aluno"
            )
        
        self._save_registry()
    
    def register_feature(self, feature: FeatureDefinition) -> None:
        """
        Registra uma nova feature.
        
        Args:
            feature: Definição da feature
        """
        self._features[feature.name] = feature
        self._save_registry()
        logger.info("feature_registered", name=feature.name)
    
    def register_group(self, group: FeatureGroup) -> None:
        """
        Registra um novo grupo de features.
        
        Args:
            group: Definição do grupo
        """
        self._groups[group.name] = group
        self._save_registry()
        logger.info("group_registered", name=group.name)
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Obtém definição de uma feature."""
        return self._features.get(name)
    
    def get_group(self, name: str) -> Optional[FeatureGroup]:
        """Obtém definição de um grupo."""
        return self._groups.get(name)
    
    def get_features_by_group(self, group_name: str) -> List[FeatureDefinition]:
        """Obtém todas as features de um grupo."""
        group = self.get_group(group_name)
        if not group:
            return []
        
        return [self._features[name] for name in group.features if name in self._features]
    
    def get_features_by_tag(self, tag: str) -> List[FeatureDefinition]:
        """Obtém features por tag."""
        return [feat for feat in self._features.values() if tag in feat.tags]
    
    def list_features(self) -> List[str]:
        """Lista nomes de todas as features."""
        return list(self._features.keys())
    
    def list_groups(self) -> List[str]:
        """Lista nomes de todos os grupos."""
        return list(self._groups.keys())
    
    def get_source_columns(self, feature_names: List[str]) -> Dict[str, str]:
        """
        Obtém mapeamento de features para colunas de origem.
        
        Args:
            feature_names: Lista de nomes de features
            
        Returns:
            Dicionário {feature_name: source_column}
        """
        mapping = {}
        for name in feature_names:
            feat = self.get_feature(name)
            if feat:
                mapping[name] = feat.source
        return mapping

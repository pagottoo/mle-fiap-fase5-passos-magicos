"""
Script principal de treinamento do modelo
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataPreprocessor, FeatureEngineer
from src.models import ModelTrainer
from src.feature_store import FeatureStore
from src.config import DATA_DIR


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento do modelo Passos Mágicos")

    parser.add_argument(
        "--data-source",
        choices=["local", "s3"],
        default=os.getenv("TRAIN_DATA_SOURCE", "local"),
        help="Origem do dataset de treino",
    )
    parser.add_argument(
        "--data-path",
        default=os.getenv("TRAIN_DATA_PATH", ""),
        help="Caminho local do dataset (quando data-source=local)",
    )
    parser.add_argument(
        "--s3-uri",
        default=os.getenv("TRAIN_S3_URI", ""),
        help="URI S3 do dataset (ex: s3://bucket/path/dataset.csv)",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        default=os.getenv("TRAIN_S3_ENDPOINT_URL", os.getenv("AWS_ENDPOINT_URL", "")),
        help="Endpoint S3 compatível (ex: MinIO). Opcional.",
    )
    parser.add_argument(
        "--local-download-path",
        default=os.getenv("TRAIN_LOCAL_DOWNLOAD_PATH", "/tmp/passos_magicos_training.csv"),
        help="Caminho local para salvar o dataset baixado do S3",
    )
    parser.add_argument(
        "--year",
        default=os.getenv("TRAIN_DATA_YEAR", "2022"),
        help="Ano usado no pré-processamento (sufixo das colunas)",
    )
    parser.add_argument(
        "--model-type",
        default=os.getenv("TRAIN_MODEL_TYPE", "random_forest"),
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        help="Tipo de modelo",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "passos-magicos-ponto-virada"),
        help="Nome do experimento MLflow",
    )
    parser.add_argument(
        "--mlflow-model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "passos-magicos-ponto-virada"),
        help="Nome do modelo no MLflow Registry",
    )

    default_promote = _env_bool("TRAIN_PROMOTE_TO_STAGING", True)
    parser.add_argument(
        "--promote-to-staging",
        dest="promote_to_staging",
        action="store_true",
        default=default_promote,
        help="Promove a última versão registrada para Staging",
    )
    parser.add_argument(
        "--no-promote-to-staging",
        dest="promote_to_staging",
        action="store_false",
        help="Não promove para Staging após registrar no MLflow",
    )

    return parser.parse_args()


def _download_from_s3(s3_uri: str, target_path: Path, endpoint_url: str = "") -> Path:
    """Baixa o dataset do S3 para caminho local."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"URI S3 inválida: {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"URI S3 sem key: {s3_uri}")

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 não instalado. Adicione boto3 nas dependências.") from exc

    client_kwargs = {}
    region = os.getenv("AWS_DEFAULT_REGION", "").strip()
    if region:
        client_kwargs["region_name"] = region
    if endpoint_url.strip():
        client_kwargs["endpoint_url"] = endpoint_url.strip()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3", **client_kwargs)
    s3_client.download_file(bucket, key, str(target_path))
    return target_path


def _resolve_data_path(args: argparse.Namespace) -> Path:
    """Resolve o dataset de treino com base na origem configurada."""
    if args.data_source == "local":
        if args.data_path:
            return Path(args.data_path)
        return DATA_DIR / "Bases antigas" / "PEDE_PASSOS_DATASET_FIAP.csv"

    # data_source == s3
    if not args.s3_uri:
        raise ValueError("TRAIN_S3_URI/--s3-uri é obrigatório quando data-source=s3")
    return _download_from_s3(
        s3_uri=args.s3_uri,
        target_path=Path(args.local_download_path),
        endpoint_url=args.s3_endpoint_url,
    )


def main():
    """Pipeline principal de treinamento."""
    args = _parse_args()

    print("=" * 60)
    print("PIPELINE DE TREINAMENTO - PASSOS MÁGICOS")
    print("Modelo de Predição de Ponto de Virada")
    print("=" * 60)

    print(f"\nConfiguração:")
    print(f" data_source: {args.data_source}")
    print(f" model_type: {args.model_type}")
    print(f" year: {args.year}")
    print(f" experiment_name: {args.experiment_name}")
    print(f" mlflow_model_name: {args.mlflow_model_name}")
    print(f" promote_to_staging: {args.promote_to_staging}")

    # 0. Inicializar Feature Store
    print("\n[0/7] Inicializando Feature Store...")
    feature_store = FeatureStore()
    print(f"Feature Store inicializada")
    print(f"Features registradas: {len(feature_store.list_features())}")

    # 1. Carregar e preparar dados
    print("\n[1/7] Carregando dados...")
    data_path = _resolve_data_path(args)
    print(f"Dataset resolvido em: {data_path}")

    preprocessor = DataPreprocessor()
    df = preprocessor.prepare_dataset(data_path, year=args.year)

    # Adicionar ID único para cada aluno
    df["aluno_id"] = range(1, len(df) + 1)

    print(f"Dados carregados: {len(df)} registros")

    # 2. Engenharia de features
    print("\n[2/7] Engenharia de features...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_target_variable(df)
    df = feature_engineer.fit_transform(df)

    X, y = feature_engineer.get_feature_matrix(df)
    print(f"Features criadas: {len(feature_engineer.feature_names)}")
    print(f"Distribuição target: {y.sum()} ponto de virada / {len(y) - y.sum()} sem ponto de virada")

    # 2.5 Ingerir dados no Feature Store (Offline)
    print("\n[2.5/7] Salvando features no Feature Store...")
    dataset_path = feature_store.ingest_training_data(
        df,
        dataset_name="passos_magicos_training",
        entity_column="aluno_id",
    )
    print(f"Dados salvos no Offline Store: {dataset_path}")

    # 3. Divisão dos dados
    print("\n[3/7] Dividindo dados...")
    trainer = ModelTrainer(
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        enable_mlflow=True,
    )

    run_started = False
    model_uri = None
    promoted_version = None

    try:
        run_name = f"k8s-{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        trainer.start_run(
            run_name=run_name,
            description=f"data_source={args.data_source};data_path={data_path};year={args.year}",
        )
        run_started = True

        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

        # 4. Validação cruzada
        print("\n[4/7] Validação cruzada...")
        cv_results = trainer.cross_validate(X_train, y_train)
        print(f"F1-Score CV: {cv_results['f1']['mean']:.4f} (+/- {cv_results['f1']['std']:.4f})")
        print(f"ROC-AUC CV: {cv_results['roc_auc']['mean']:.4f} (+/- {cv_results['roc_auc']['std']:.4f})")

        # 5. Treinamento final
        print("\n[5/7] Treinamento final...")
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        # 6. Salvar modelo
        print("\n[6/7] Salvando modelo...")
        model_path = trainer.save_model(preprocessor, feature_engineer)
        print(f"Modelo salvo em: {model_path}")

        # Registrar no MLflow Registry
        print("\n[6.5/7] Registrando modelo no MLflow...")
        model_uri = trainer.log_model_to_mlflow(
            X_sample=X_test[:5],
            y_sample=y_test[:5],
            registered_model_name=args.mlflow_model_name,
        )
        if model_uri:
            print(f"Modelo registrado no MLflow: {model_uri}")
        else:
            print("Modelo não registrado no MLflow (verifique MLFLOW_ENABLED e MLFLOW_TRACKING_URI)")

        if args.promote_to_staging and trainer.model_registry:
            versions = trainer.model_registry.get_latest_versions(args.mlflow_model_name)
            if versions:
                promoted_version = int(versions[0].version)
                trainer.model_registry.promote_to_staging(args.mlflow_model_name, promoted_version)
                print(f"Versão {promoted_version} promovida para Staging")
            else:
                print("Nenhuma versão encontrada para promoção em Staging")

        # 7. Materializar features no Online Store (para inferência)
        print("\n[7/7] Materializando features para inferência...")
        count = feature_store.materialize_for_serving(
            df,
            table_name="alunos_features",
            entity_column="aluno_id",
        )
        print(f"{count} registros materializados no Online Store")

        # Sincronizar Offline -> Online para garantir consistência
        feature_store.sync_offline_to_online(
            dataset_name="passos_magicos_training",
            table_name="alunos_inference",
            entity_column="aluno_id",
        )
        print(f"Dados sincronizados Offline -> Online")

    except Exception:
        if run_started:
            trainer.end_run(status="FAILED")
        raise
    else:
        if run_started:
            trainer.end_run(status="FINISHED")

    # Resumo do modelo
    print("\n" + "=" * 60)
    print(trainer.get_model_summary())

    # Status do Feature Store
    print("\n" + "=" * 60)
    print("STATUS DO FEATURE STORE")
    print("=" * 60)
    status = feature_store.get_status()
    print(f"Datasets offline: {status['offline_store']['datasets']}")
    print(f"Tabelas online: {status['online_store']['tables']}")

    print("\nOutputs:")
    print(f"mlflow_model_uri: {model_uri}")
    print(f"promoted_staging_version: {promoted_version}")
    print("\n Pipeline de treinamento concluído com sucesso!")

    return trainer, preprocessor, feature_engineer, feature_store


if __name__ == "__main__":
    main()

"""
Main model training script.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from structlog.contextvars import bind_contextvars, clear_contextvars

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR
from src.data import DataPreprocessor, FeatureEngineer
from src.feature_store import FeatureStore
from src.models import ModelTrainer
from src.monitoring.job_metrics import JobMetricsPusher
from src.monitoring.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(component="training_job")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passos Magicos model training")

    parser.add_argument(
        "--data-source",
        choices=["local", "s3"],
        default=os.getenv("TRAIN_DATA_SOURCE", "local"),
        help="Training dataset source",
    )
    parser.add_argument(
        "--data-path",
        default=os.getenv("TRAIN_DATA_PATH", ""),
        help="Local dataset path (when data-source=local)",
    )
    parser.add_argument(
        "--s3-uri",
        default=os.getenv("TRAIN_S3_URI", ""),
        help="S3 dataset URI (e.g. s3://bucket/path/dataset.csv)",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        default=os.getenv("TRAIN_S3_ENDPOINT_URL", os.getenv("AWS_ENDPOINT_URL", "")),
        help="S3-compatible endpoint (e.g. MinIO). Optional.",
    )
    parser.add_argument(
        "--local-download-path",
        default=os.getenv("TRAIN_LOCAL_DOWNLOAD_PATH", "/tmp/passos_magicos_training.csv"),
        help="Local destination path for S3 download",
    )
    parser.add_argument(
        "--year",
        default=os.getenv("TRAIN_DATA_YEAR", "2022"),
        help="Year suffix used during preprocessing",
    )
    parser.add_argument(
        "--model-type",
        default=os.getenv("TRAIN_MODEL_TYPE", "random_forest"),
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        help="Model type",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "passos-magicos-ponto-virada"),
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "passos-magicos-ponto-virada"),
        help="Model name in MLflow Registry",
    )

    default_promote = _env_bool("TRAIN_PROMOTE_TO_STAGING", True)
    parser.add_argument(
        "--promote-to-staging",
        dest="promote_to_staging",
        action="store_true",
        default=default_promote,
        help="Promote latest registered version to Staging",
    )
    parser.add_argument(
        "--no-promote-to-staging",
        dest="promote_to_staging",
        action="store_false",
        help="Do not promote to Staging after MLflow registration",
    )

    return parser.parse_args()


def _download_from_s3(s3_uri: str, target_path: Path, endpoint_url: str = "") -> Path:
    """Download dataset from S3 to local path."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI has no object key: {s3_uri}")

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed. Add boto3 to dependencies.") from exc

    client_kwargs = {}
    region = os.getenv("AWS_DEFAULT_REGION", "").strip()
    if region:
        client_kwargs["region_name"] = region
    if endpoint_url.strip():
        client_kwargs["endpoint_url"] = endpoint_url.strip()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "training_data_download_started",
        source="s3",
        s3_bucket=bucket,
        s3_key=key,
        target_path=str(target_path),
        s3_endpoint=endpoint_url.strip() or None,
    )

    s3_client = boto3.client("s3", **client_kwargs)
    s3_client.download_file(bucket, key, str(target_path))

    logger.info(
        "training_data_download_completed",
        source="s3",
        target_path=str(target_path),
    )
    return target_path


def _resolve_data_path(args: argparse.Namespace) -> Path:
    """Resolve training dataset path based on configured source."""
    if args.data_source == "local":
        if args.data_path:
            return Path(args.data_path)
        return DATA_DIR / "Bases antigas" / "PEDE_PASSOS_DATASET_FIAP.csv"

    if not args.s3_uri:
        raise ValueError("TRAIN_S3_URI/--s3-uri is required when data-source=s3")

    return _download_from_s3(
        s3_uri=args.s3_uri,
        target_path=Path(args.local_download_path),
        endpoint_url=args.s3_endpoint_url,
    )


def main():
    """Main training pipeline."""
    started_at = time.time()
    metrics_pusher = JobMetricsPusher(component="training")
    bind_contextvars(run_id=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    exit_code = 0
    success = False

    trainer = None
    preprocessor = None
    feature_engineer = None
    feature_store = None

    run_started = False
    run_id = None
    model_uri = None
    promoted_version = None
    train_metrics = {}

    args = _parse_args()
    logger.info(
        "training_job_started",
        data_source=args.data_source,
        model_type=args.model_type,
        year=args.year,
        experiment_name=args.experiment_name,
        mlflow_model_name=args.mlflow_model_name,
        promote_to_staging=args.promote_to_staging,
    )

    try:
        logger.info("training_step_started", step="feature_store_init", step_order="0/7")
        feature_store = FeatureStore()
        logger.info(
            "training_step_completed",
            step="feature_store_init",
            features_registered=len(feature_store.list_features()),
        )

        logger.info("training_step_started", step="load_data", step_order="1/7")
        data_path = _resolve_data_path(args)
        logger.info("training_dataset_resolved", data_path=str(data_path))

        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_dataset(data_path, year=args.year)
        df["aluno_id"] = range(1, len(df) + 1)

        logger.info(
            "training_step_completed",
            step="load_data",
            records=len(df),
        )

        logger.info("training_step_started", step="feature_engineering", step_order="2/7")
        feature_engineer = FeatureEngineer()
        df = feature_engineer.create_target_variable(df)
        df = feature_engineer.fit_transform(df)

        X, y = feature_engineer.get_feature_matrix(df)
        turning_point_count = int(y.sum())
        logger.info(
            "training_step_completed",
            step="feature_engineering",
            features_created=len(feature_engineer.feature_names),
            records=len(df),
            turning_point_count=turning_point_count,
            no_turning_point_count=int(len(y) - turning_point_count),
        )

        logger.info("training_step_started", step="offline_store_ingest", step_order="2.5/7")
        dataset_path = feature_store.ingest_training_data(
            df,
            dataset_name="passos_magicos_training",
            entity_column="aluno_id",
        )
        logger.info(
            "training_step_completed",
            step="offline_store_ingest",
            offline_dataset_path=str(dataset_path),
        )

        logger.info("training_step_started", step="split_and_train_setup", step_order="3/7")
        trainer = ModelTrainer(
            model_type=args.model_type,
            experiment_name=args.experiment_name,
            enable_mlflow=True,
        )

        run_name = f"k8s-{args.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        trainer.start_run(
            run_name=run_name,
            description=f"data_source={args.data_source};data_path={data_path};year={args.year}",
        )
        run_started = True
        run_id = trainer.run_id
        if run_id:
            bind_contextvars(run_id=run_id)

        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        logger.info(
            "training_step_completed",
            step="split_and_train_setup",
            run_name=run_name,
            run_id=run_id,
            train_samples=len(X_train),
            test_samples=len(X_test),
        )

        logger.info("training_step_started", step="cross_validation", step_order="4/7")
        cv_results = trainer.cross_validate(X_train, y_train)
        logger.info(
            "training_step_completed",
            step="cross_validation",
            cv_f1_mean=cv_results["f1"]["mean"],
            cv_f1_std=cv_results["f1"]["std"],
            cv_roc_auc_mean=cv_results["roc_auc"]["mean"],
            cv_roc_auc_std=cv_results["roc_auc"]["std"],
        )

        logger.info("training_step_started", step="final_training", step_order="5/7")
        trainer.train(X_train, y_train)
        train_metrics = trainer.evaluate(X_test, y_test)
        logger.info(
            "training_step_completed",
            step="final_training",
            accuracy=train_metrics.get("accuracy"),
            precision=train_metrics.get("precision"),
            recall=train_metrics.get("recall"),
            f1_score=train_metrics.get("f1_score"),
            roc_auc=train_metrics.get("roc_auc"),
        )

        logger.info("training_step_started", step="save_model", step_order="6/7")
        model_path = trainer.save_model(preprocessor, feature_engineer)
        logger.info(
            "training_step_completed",
            step="save_model",
            model_path=str(model_path),
        )

        logger.info("training_step_started", step="mlflow_register", step_order="6.5/7")
        model_uri = trainer.log_model_to_mlflow(
            X_sample=X_test[:5],
            y_sample=y_test[:5],
            registered_model_name=args.mlflow_model_name,
        )
        if model_uri:
            logger.info(
                "training_step_completed",
                step="mlflow_register",
                model_uri=model_uri,
                mlflow_model_name=args.mlflow_model_name,
            )
        else:
            logger.warning(
                "training_step_warning",
                step="mlflow_register",
                reason="model_not_registered",
                mlflow_enabled="verify_MLFLOW_ENABLED_and_MLFLOW_TRACKING_URI",
            )

        if args.promote_to_staging and trainer.model_registry:
            versions = trainer.model_registry.get_latest_versions(args.mlflow_model_name)
            if versions:
                promoted_version = int(versions[0].version)
                trainer.model_registry.promote_to_staging(args.mlflow_model_name, promoted_version)
                logger.info(
                    "training_model_promoted_to_staging",
                    mlflow_model_name=args.mlflow_model_name,
                    promoted_version=promoted_version,
                )
                bind_contextvars(model_version=str(promoted_version))
            else:
                logger.warning(
                    "training_model_promote_skipped",
                    reason="no_versions_found",
                    mlflow_model_name=args.mlflow_model_name,
                )

        logger.info("training_step_started", step="materialize_online_store", step_order="7/7")
        count = feature_store.materialize_for_serving(
            df,
            table_name="alunos_features",
            entity_column="aluno_id",
        )

        feature_store.sync_offline_to_online(
            dataset_name="passos_magicos_training",
            table_name="alunos_inference",
            entity_column="aluno_id",
        )
        logger.info(
            "training_step_completed",
            step="materialize_online_store",
            online_records=count,
            online_table="alunos_features",
            sync_table="alunos_inference",
        )

        success = True

    except Exception as exc:
        exit_code = 1
        logger.error(
            "training_job_failed",
            error=str(exc),
            run_id=run_id,
            exc_info=True,
        )

        if run_started and trainer is not None:
            trainer.end_run(status="FAILED")
            logger.info(
                "training_mlflow_run_ended",
                run_id=run_id,
                status="FAILED",
            )
        raise

    else:
        if run_started and trainer is not None:
            trainer.end_run(status="FINISHED")
            logger.info(
                "training_mlflow_run_ended",
                run_id=run_id,
                status="FINISHED",
            )

        if feature_store is not None:
            status = feature_store.get_status()
            logger.info(
                "training_feature_store_status",
                datasets_offline=status["offline_store"].get("datasets", []),
                tables_online=status["online_store"].get("tables", []),
            )

        if trainer is not None:
            logger.info(
                "training_model_summary_generated",
                summary=trainer.get_model_summary(),
            )

        logger.info(
            "training_job_completed",
            model_uri=model_uri,
            promoted_staging_version=promoted_version,
            run_id=run_id,
        )

    finally:
        duration = time.time() - started_at
        extra_metrics = {
            "training_last_f1_score": float(train_metrics.get("f1_score", 0.0) or 0.0),
            "training_last_roc_auc": float(train_metrics.get("roc_auc", 0.0) or 0.0),
            "training_last_registered_model": 1.0 if model_uri else 0.0,
            "training_last_promoted_version": float(promoted_version or 0),
        }

        metrics_pusher.push_run_metrics(
            success=success,
            duration_seconds=duration,
            exit_code=exit_code,
            extra_metrics=extra_metrics,
        )

        logger.info(
            "training_job_final_status",
            success=success,
            exit_code=exit_code,
            duration_seconds=round(duration, 4),
            run_id=run_id,
        )
        clear_contextvars()

    return trainer, preprocessor, feature_engineer, feature_store


if __name__ == "__main__":
    main()

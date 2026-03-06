"""
Model Lifecycle Operations Utility.
Used by CI/CD workflows to interact with MLflow Registry.
"""
import argparse
import sys
import logging
from typing import Optional

# Ensure project root is in path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mlflow_tracking import ModelRegistry
from src.monitoring.logger import setup_logging, get_logger

# Force logging to stderr for CLI tools that use stdout for GHA outputs
def setup_cli_logging():
    setup_logging()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setStream(sys.stderr)

setup_cli_logging()
logger = get_logger(component="model_ops")

def check_versions(model_name: str):
    """Print current versions for GHA outputs."""
    registry = ModelRegistry()
    
    prod = registry.get_production_version(model_name)
    staging = registry.get_staging_version(model_name)
    
    prod_v = prod.version if prod else "None"
    staging_v = staging.version if staging else "None"
    
    # We output in a format that GHA can parse easily (stdout only for these)
    sys.stdout.write(f"production_version={prod_v}\n")
    sys.stdout.write(f"staging_version={staging_v}\n")
    
    # Logic for target version
    if staging_v != "None":
        sys.stdout.write(f"target_version={staging_v}\n")
    else:
        sys.stdout.write(f"target_version=None\n")
    sys.stdout.flush()

def promote_model(model_name: str, version: int, stage: str):
    """Promote a specific version to a stage."""
    registry = ModelRegistry()
    logger.info("promoting_model", name=model_name, version=version, to_stage=stage)
    
    if stage.lower() == "staging":
        registry.promote_to_staging(model_name, version)
    elif stage.lower() == "production":
        registry.promote_to_production(model_name, version)
    else:
        sys.stderr.write(f"Error: Invalid stage {stage}\n")
        sys.exit(1)
    
    sys.stderr.write(f"Success: Model {model_name} v{version} promoted to {stage}\n")

def main():
    parser = argparse.ArgumentParser(description="MLOps Model Lifecycle CLI")
    parser.add_argument("--action", choices=["check", "promote"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", type=int, help="Version number for promotion")
    parser.add_argument("--stage", choices=["Staging", "Production"], help="Target stage")
    
    args = parser.parse_args()
    
    try:
        if args.action == "check":
            check_versions(args.model_name)
        elif args.action == "promote":
            if not args.version or not args.stage:
                print("Error: --version and --stage are required for promote action")
                sys.exit(1)
            promote_model(args.model_name, args.version, args.stage)
    except Exception as e:
        logger.error("ops_failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()

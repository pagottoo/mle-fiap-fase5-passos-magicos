"""
Testes para utilitários de logging estruturado.
"""
from unittest.mock import Mock

import src.monitoring.logger as logger_module


class TestMonitoringLoggerExtended:
    def test_setup_logging_calls_structlog_configure(self, monkeypatch):
        configure_mock = Mock()
        monkeypatch.setattr(logger_module.structlog, "configure", configure_mock)

        logger_module.setup_logging()

        assert configure_mock.called

    def test_log_prediction_and_get_recent_predictions(self, monkeypatch, tmp_path):
        monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)

        fake_logger = Mock()
        monkeypatch.setattr(logger_module.structlog, "get_logger", Mock(return_value=fake_logger))

        assert logger_module.get_recent_predictions() == []

        logger_module.log_prediction({"aluno_id": 1}, {"prediction": 1, "confidence": 0.9})
        logger_module.log_prediction({"aluno_id": 2}, {"prediction": 0, "confidence": 0.8})

        fake_logger.info.assert_called()

        recent = logger_module.get_recent_predictions(n=1)
        assert len(recent) == 1
        assert recent[0]["input"]["aluno_id"] == 2
        assert recent[0]["output"]["prediction"] == 0

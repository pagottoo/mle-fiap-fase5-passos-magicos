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

    def test_log_feedback_and_get_recent_feedback(self, monkeypatch, tmp_path):
        from unittest.mock import ANY
        monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)
        fake_logger = Mock()
        monkeypatch.setattr(logger_module.structlog, "get_logger", Mock(return_value=fake_logger))

        # Test empty
        assert logger_module.get_recent_feedback() == []

        feedback_data = {"prediction_id": "uuid-1", "actual_outcome": 1, "was_correct": True}
        logger_module.log_feedback(feedback_data)

        # Check logger call with ANY timestamp
        fake_logger.info.assert_called_with(
            "feedback_logged", 
            type="feedback", 
            timestamp=ANY,
            **feedback_data
        )
        
        recent = logger_module.get_recent_feedback(n=10)
        assert len(recent) == 1
        assert recent[0]["prediction_id"] == "uuid-1"
        assert recent[0]["was_correct"] is True

    def test_get_recent_predictions_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path / "nonexistent")
        assert logger_module.get_recent_predictions() == []

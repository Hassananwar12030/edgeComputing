"""
Basic Tests
===========

Sanity checks to ensure imports work.
"""

import pytest


def test_import_edge():
    """Test that edge module can be imported."""
    from src import edge
    assert hasattr(edge, 'EdgeNode')


def test_import_server():
    """Test that server module can be imported."""
    from src import server
    assert hasattr(server, 'Server')


def test_import_common():
    """Test that common module can be imported."""
    from src import common
    assert hasattr(common, 'load_config')


def test_constants():
    """Test that constants are defined."""
    from src.common.constants import (
        AUDIO_SAMPLE_RATE,
        FRAME_WIDTH,
        STRATEGIES
    )

    assert AUDIO_SAMPLE_RATE == 16000
    assert FRAME_WIDTH == 640
    assert len(STRATEGIES) == 3


def test_spectrogram_shape():
    """Test expected spectrogram shape."""
    from src.common.constants import (
        AUDIO_SAMPLE_RATE,
        AUDIO_CHUNK_DURATION,
        STFT_HOP_LENGTH,
        STFT_WINDOW_LENGTH,
        STFT_N_MELS
    )

    # Calculate expected time frames
    chunk_samples = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION)
    n_frames = 1 + (chunk_samples - STFT_WINDOW_LENGTH) // STFT_HOP_LENGTH

    assert n_frames == 49
    assert STFT_N_MELS == 128


class TestSTFTProcessor:
    """Tests for STFT processing."""

    def test_stft_import(self):
        """Test STFT processor can be imported."""
        from src.edge.processing.stft import STFTProcessor
        assert STFTProcessor is not None

    def test_stft_initialization(self):
        """Test STFT processor initialization."""
        from src.edge.processing.stft import STFTProcessor

        processor = STFTProcessor()
        assert processor.sample_rate == 16000
        assert processor.n_mels == 128


class TestStrategies:
    """Tests for learning strategies."""

    def test_strategy_imports(self):
        """Test all strategies can be imported."""
        from src.edge.strategies import (
            CentralizedStrategy,
            HybridStrategy,
            FederatedStrategy
        )

        assert CentralizedStrategy is not None
        assert HybridStrategy is not None
        assert FederatedStrategy is not None

    def test_strategy_factory(self):
        """Test strategy factory."""
        from src.edge.strategies import StrategyFactory

        # Test valid strategies
        for strategy_type in ["centralized", "hybrid", "federated"]:
            strategy = StrategyFactory.create(strategy_type, node_id="test")
            assert strategy.name == strategy_type

    def test_invalid_strategy(self):
        """Test factory raises on invalid strategy."""
        from src.edge.strategies import StrategyFactory

        with pytest.raises(ValueError):
            StrategyFactory.create("invalid")

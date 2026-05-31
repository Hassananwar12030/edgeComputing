"""
Test Suite
==========

Test structure:
    tests/
    ├── edge/           # Edge module tests
    │   ├── test_sensors.py
    │   ├── test_processing.py
    │   └── test_strategies.py
    ├── server/         # Server module tests
    │   ├── test_flower.py
    │   └── test_training.py
    ├── common/         # Common module tests
    │   └── test_utils.py
    └── integration/    # End-to-end tests
        └── test_pipeline.py

Running tests:
    pytest tests/
    pytest tests/edge/
    pytest tests/edge/test_stft.py -v
"""

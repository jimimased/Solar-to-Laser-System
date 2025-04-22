"""
Data Collection Module for the Solar-to-Laser System.

This module provides components for collecting data from solar panels,
normalizing it, and storing it in a database.
"""

from .collector import (
    SensorInterface,
    ArduinoSensorInterface,
    RaspberryPiSensorInterface,
    SimulatedSensorInterface,
    WeatherDataProvider,
    SolarDataCollector,
    normalize_voltage,
    normalize_current,
    calculate_power,
    filter_outliers,
)

from .storage import (
    StorageInterface,
    InfluxDBStorage,
    PostgreSQLStorage,
    FileStorage,
)

from .api import (
    create_api,
    api,
    SolarDataModel,
    SolarDataBatch,
    StatisticsResponse,
    TimeSeriesResponse,
    StatusResponse,
)

__all__ = [
    # Collector
    'SensorInterface',
    'ArduinoSensorInterface',
    'RaspberryPiSensorInterface',
    'SimulatedSensorInterface',
    'WeatherDataProvider',
    'SolarDataCollector',
    'normalize_voltage',
    'normalize_current',
    'calculate_power',
    'filter_outliers',
    
    # Storage
    'StorageInterface',
    'InfluxDBStorage',
    'PostgreSQLStorage',
    'FileStorage',
    
    # API
    'create_api',
    'api',
    'SolarDataModel',
    'SolarDataBatch',
    'StatisticsResponse',
    'TimeSeriesResponse',
    'StatusResponse',
]
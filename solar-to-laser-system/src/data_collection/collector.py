"""
Solar data collector implementation.

This module provides classes for collecting data from solar panels.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

import requests
import paho.mqtt.client as mqtt

from ..common import SolarData

logger = logging.getLogger(__name__)


class SensorInterface:
    """Base class for sensor interfaces."""
    
    def read(self) -> Dict[str, float]:
        """Read sensor values.
        
        Returns:
            Dict[str, float]: Dictionary of sensor values
        """
        raise NotImplementedError("Subclasses must implement read()")


class ArduinoSensorInterface(SensorInterface):
    """Interface for Arduino-based sensors."""
    
    def __init__(self, port: str, baud_rate: int = 9600):
        """Initialize the Arduino sensor interface.
        
        Args:
            port: Serial port
            baud_rate: Baud rate
        """
        self.port = port
        self.baud_rate = baud_rate
        self._serial = None
        self._connect()
    
    def _connect(self):
        """Connect to the Arduino."""
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud_rate)
            logger.info(f"Connected to Arduino on {self.port}")
        except ImportError:
            logger.error("pyserial not installed. Install with 'pip install pyserial'")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            raise
    
    def read(self) -> Dict[str, float]:
        """Read sensor values from Arduino.
        
        Returns:
            Dict[str, float]: Dictionary with voltage, current, and power values
        """
        if not self._serial:
            self._connect()
        
        try:
            # Send command to request data
            self._serial.write(b'READ\n')
            
            # Read response
            response = self._serial.readline().decode('utf-8').strip()
            
            # Parse response (format: "voltage:X,current:Y,power:Z")
            values = {}
            for pair in response.split(','):
                key, value = pair.split(':')
                values[key.strip()] = float(value.strip())
            
            return values
        except Exception as e:
            logger.error(f"Error reading from Arduino: {e}")
            return {"voltage": 0.0, "current": 0.0, "power": 0.0}


class RaspberryPiSensorInterface(SensorInterface):
    """Interface for Raspberry Pi GPIO-based sensors."""
    
    def __init__(self, voltage_pin: int, current_pin: int):
        """Initialize the Raspberry Pi sensor interface.
        
        Args:
            voltage_pin: GPIO pin for voltage sensor
            current_pin: GPIO pin for current sensor
        """
        self.voltage_pin = voltage_pin
        self.current_pin = current_pin
        self._setup_gpio()
    
    def _setup_gpio(self):
        """Set up GPIO pins."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.voltage_pin, GPIO.IN)
            GPIO.setup(self.current_pin, GPIO.IN)
            logger.info(f"Set up GPIO pins: voltage={self.voltage_pin}, current={self.current_pin}")
        except ImportError:
            logger.error("RPi.GPIO not installed or not running on a Raspberry Pi")
            raise
        except Exception as e:
            logger.error(f"Failed to set up GPIO: {e}")
            raise
    
    def read(self) -> Dict[str, float]:
        """Read sensor values from GPIO pins.
        
        Returns:
            Dict[str, float]: Dictionary with voltage, current, and power values
        """
        try:
            import RPi.GPIO as GPIO
            
            # Read analog values (using ADC would be required in real implementation)
            # This is a simplified example
            voltage_raw = GPIO.input(self.voltage_pin)
            current_raw = GPIO.input(self.current_pin)
            
            # Convert raw values to actual measurements
            # In a real implementation, this would involve proper calibration
            voltage = voltage_raw * 5.0  # Example conversion
            current = current_raw * 1.0  # Example conversion
            power = voltage * current
            
            return {
                "voltage": voltage,
                "current": current,
                "power": power
            }
        except Exception as e:
            logger.error(f"Error reading from GPIO: {e}")
            return {"voltage": 0.0, "current": 0.0, "power": 0.0}


class SimulatedSensorInterface(SensorInterface):
    """Simulated sensor interface for testing and development."""
    
    def __init__(
        self,
        voltage_range: Tuple[float, float] = (0.0, 48.0),
        current_range: Tuple[float, float] = (0.0, 10.0),
        simulation_mode: str = "sine"
    ):
        """Initialize the simulated sensor interface.
        
        Args:
            voltage_range: Range of voltage values (min, max)
            current_range: Range of current values (min, max)
            simulation_mode: Simulation mode ("sine", "random", "step")
        """
        self.voltage_range = voltage_range
        self.current_range = current_range
        self.simulation_mode = simulation_mode
        self.start_time = time.time()
    
    def read(self) -> Dict[str, float]:
        """Generate simulated sensor values.
        
        Returns:
            Dict[str, float]: Dictionary with voltage, current, and power values
        """
        elapsed_time = time.time() - self.start_time
        
        if self.simulation_mode == "sine":
            import math
            # Sine wave simulation
            voltage = (
                (self.voltage_range[1] - self.voltage_range[0]) / 2 * 
                math.sin(elapsed_time * 0.1) + 
                (self.voltage_range[1] + self.voltage_range[0]) / 2
            )
            current = (
                (self.current_range[1] - self.current_range[0]) / 2 * 
                math.sin(elapsed_time * 0.05) + 
                (self.current_range[1] + self.current_range[0]) / 2
            )
        elif self.simulation_mode == "random":
            import random
            # Random values within range
            voltage = random.uniform(*self.voltage_range)
            current = random.uniform(*self.current_range)
        elif self.simulation_mode == "step":
            # Step function
            step_interval = 10  # seconds
            step = int(elapsed_time / step_interval) % 5
            voltage_step = (self.voltage_range[1] - self.voltage_range[0]) / 4
            current_step = (self.current_range[1] - self.current_range[0]) / 4
            voltage = self.voltage_range[0] + step * voltage_step
            current = self.current_range[0] + step * current_step
        else:
            # Default to constant values
            voltage = (self.voltage_range[0] + self.voltage_range[1]) / 2
            current = (self.current_range[0] + self.current_range[1]) / 2
        
        power = voltage * current
        
        return {
            "voltage": voltage,
            "current": current,
            "power": power
        }


class WeatherDataProvider:
    """Provider for weather data to supplement solar data."""
    
    def __init__(self, api_key: str, location: str):
        """Initialize the weather data provider.
        
        Args:
            api_key: API key for weather service
            location: Location for weather data
        """
        self.api_key = api_key
        self.location = location
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather_data(self) -> Dict[str, Any]:
        """Get current weather data.
        
        Returns:
            Dict[str, Any]: Weather data
        """
        try:
            params = {
                "q": self.location,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "cloud_cover": data["clouds"]["all"],
                "wind_speed": data["wind"]["speed"],
                "weather_condition": data["weather"][0]["main"],
                "irradiance": self._estimate_irradiance(data)
            }
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {
                "temperature": 25.0,
                "humidity": 50.0,
                "cloud_cover": 0.0,
                "wind_speed": 0.0,
                "weather_condition": "Clear",
                "irradiance": 1000.0
            }
    
    def _estimate_irradiance(self, weather_data: Dict[str, Any]) -> float:
        """Estimate solar irradiance based on weather data.
        
        Args:
            weather_data: Weather data from API
        
        Returns:
            float: Estimated irradiance in W/m²
        """
        # This is a simplified model for estimation
        # A more accurate model would consider location, time of day, season, etc.
        
        # Base irradiance on a clear day
        base_irradiance = 1000.0  # W/m²
        
        # Adjust for cloud cover
        cloud_factor = 1.0 - (weather_data["clouds"]["all"] / 100.0) * 0.7
        
        # Adjust for weather condition
        condition = weather_data["weather"][0]["main"].lower()
        condition_factors = {
            "clear": 1.0,
            "few clouds": 0.9,
            "scattered clouds": 0.8,
            "broken clouds": 0.7,
            "overcast": 0.5,
            "mist": 0.6,
            "fog": 0.4,
            "rain": 0.3,
            "thunderstorm": 0.2,
            "snow": 0.3,
            "drizzle": 0.5
        }
        condition_factor = condition_factors.get(condition, 0.7)
        
        # Calculate estimated irradiance
        irradiance = base_irradiance * cloud_factor * condition_factor
        
        return irradiance


class SolarDataCollector:
    """Collector for solar panel data."""
    
    def __init__(
        self,
        sensor_interface: SensorInterface,
        weather_provider: Optional[WeatherDataProvider] = None,
        collection_interval: float = 1.0,
        preprocessing_functions: Optional[List[Callable[[Dict[str, float]], Dict[str, float]]]] = None
    ):
        """Initialize the solar data collector.
        
        Args:
            sensor_interface: Interface for reading sensor data
            weather_provider: Provider for weather data
            collection_interval: Interval between data collections in seconds
            preprocessing_functions: Functions for preprocessing sensor data
        """
        self.sensor_interface = sensor_interface
        self.weather_provider = weather_provider
        self.collection_interval = collection_interval
        self.preprocessing_functions = preprocessing_functions or []
        self.mqtt_client = None
        self.running = False
    
    def collect_single(self) -> SolarData:
        """Collect a single data point.
        
        Returns:
            SolarData: Collected data
        """
        # Read sensor values
        sensor_data = self.sensor_interface.read()
        
        # Apply preprocessing functions
        for func in self.preprocessing_functions:
            sensor_data = func(sensor_data)
        
        # Get weather data if available
        metadata = {}
        if self.weather_provider:
            weather_data = self.weather_provider.get_weather_data()
            metadata["weather"] = weather_data
            irradiance = weather_data.get("irradiance")
        else:
            irradiance = None
        
        # Create SolarData object
        solar_data = SolarData(
            timestamp=datetime.now(),
            voltage=sensor_data["voltage"],
            current=sensor_data["current"],
            power=sensor_data["power"],
            temperature=sensor_data.get("temperature"),
            irradiance=irradiance,
            metadata=metadata
        )
        
        return solar_data
    
    def start_collection(self, callback: Callable[[SolarData], None]):
        """Start continuous data collection.
        
        Args:
            callback: Function to call with each collected data point
        """
        self.running = True
        
        try:
            while self.running:
                # Collect data
                solar_data = self.collect_single()
                
                # Call callback with collected data
                callback(solar_data)
                
                # Wait for next collection
                time.sleep(self.collection_interval)
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user")
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
        finally:
            self.running = False
    
    def stop_collection(self):
        """Stop continuous data collection."""
        self.running = False
    
    def setup_mqtt_publishing(self, broker: str, topic: str, client_id: str = "solar_collector"):
        """Set up MQTT publishing for collected data.
        
        Args:
            broker: MQTT broker address
            topic: MQTT topic to publish to
            client_id: MQTT client ID
        """
        self.mqtt_client = mqtt.Client(client_id)
        
        try:
            self.mqtt_client.connect(broker)
            logger.info(f"Connected to MQTT broker at {broker}")
            
            # Start the MQTT loop in a background thread
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            self.mqtt_client = None
    
    def publish_to_mqtt(self, topic: str, solar_data: SolarData):
        """Publish solar data to MQTT.
        
        Args:
            topic: MQTT topic to publish to
            solar_data: Solar data to publish
        """
        if not self.mqtt_client:
            logger.warning("MQTT client not set up. Call setup_mqtt_publishing() first.")
            return
        
        try:
            # Convert SolarData to JSON
            import json
            payload = json.dumps(solar_data.to_dict())
            
            # Publish to MQTT
            self.mqtt_client.publish(topic, payload)
        except Exception as e:
            logger.error(f"Error publishing to MQTT: {e}")


# Preprocessing functions

def normalize_voltage(data: Dict[str, float], min_voltage: float = 0.0, max_voltage: float = 48.0) -> Dict[str, float]:
    """Normalize voltage to the range [0, 1].
    
    Args:
        data: Sensor data
        min_voltage: Minimum expected voltage
        max_voltage: Maximum expected voltage
    
    Returns:
        Dict[str, float]: Data with normalized voltage
    """
    voltage = data["voltage"]
    normalized = (voltage - min_voltage) / (max_voltage - min_voltage)
    normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
    
    result = data.copy()
    result["voltage_raw"] = voltage
    result["voltage"] = normalized
    
    return result


def normalize_current(data: Dict[str, float], min_current: float = 0.0, max_current: float = 10.0) -> Dict[str, float]:
    """Normalize current to the range [0, 1].
    
    Args:
        data: Sensor data
        min_current: Minimum expected current
        max_current: Maximum expected current
    
    Returns:
        Dict[str, float]: Data with normalized current
    """
    current = data["current"]
    normalized = (current - min_current) / (max_current - min_current)
    normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
    
    result = data.copy()
    result["current_raw"] = current
    result["current"] = normalized
    
    return result


def calculate_power(data: Dict[str, float]) -> Dict[str, float]:
    """Calculate power from voltage and current.
    
    Args:
        data: Sensor data
    
    Returns:
        Dict[str, float]: Data with calculated power
    """
    # Use raw values if available, otherwise use normalized values
    voltage = data.get("voltage_raw", data["voltage"])
    current = data.get("current_raw", data["current"])
    
    power = voltage * current
    
    result = data.copy()
    result["power"] = power
    
    return result


def filter_outliers(data: Dict[str, float], key: str, z_threshold: float = 3.0) -> Dict[str, float]:
    """Filter outliers using Z-score.
    
    Args:
        data: Sensor data
        key: Key to filter
        z_threshold: Z-score threshold
    
    Returns:
        Dict[str, float]: Data with outliers filtered
    """
    # This function would need to maintain a history of values
    # to calculate mean and standard deviation
    # This is a simplified version that doesn't actually filter
    
    return data
"""
Tests for the data collection module.
"""

import unittest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.common.data_structures import SolarData
from src.data_collection.collector import SolarDataCollector
from src.data_collection.storage import InfluxDBStorage


class TestSolarDataCollector(unittest.TestCase):
    """Tests for the SolarDataCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_storage = MagicMock()
        self.collector = SolarDataCollector(storage=self.mock_storage)
    
    def test_collect_data(self):
        """Test collecting data from a solar panel."""
        # Mock the solar panel data
        mock_data = {
            "voltage": 12.5,
            "current": 2.1,
            "power": 26.25,
            "temperature": 25.0,
            "irradiance": 800.0
        }
        
        # Mock the _read_sensor method to return the mock data
        with patch.object(self.collector, '_read_sensor', return_value=mock_data):
            # Call the collect_data method
            result = self.collector.collect_data(panel_id="panel1")
            
            # Assert that the result is a SolarData object with the correct values
            self.assertIsInstance(result, SolarData)
            self.assertEqual(result.voltage, 12.5)
            self.assertEqual(result.current, 2.1)
            self.assertEqual(result.power, 26.25)
            self.assertEqual(result.temperature, 25.0)
            self.assertEqual(result.irradiance, 800.0)
    
    def test_store_data(self):
        """Test storing data in the storage backend."""
        # Create a SolarData object
        data = SolarData(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Call the store_data method
        self.collector.store_data(data)
        
        # Assert that the storage.store method was called with the correct data
        self.mock_storage.store.assert_called_once_with(data)
    
    def test_collect_and_store(self):
        """Test collecting and storing data in one operation."""
        # Mock the solar panel data
        mock_data = {
            "voltage": 12.5,
            "current": 2.1,
            "power": 26.25,
            "temperature": 25.0,
            "irradiance": 800.0
        }
        
        # Mock the _read_sensor method to return the mock data
        with patch.object(self.collector, '_read_sensor', return_value=mock_data):
            # Call the collect_and_store method
            result = self.collector.collect_and_store(panel_id="panel1")
            
            # Assert that the result is a SolarData object with the correct values
            self.assertIsInstance(result, SolarData)
            self.assertEqual(result.voltage, 12.5)
            self.assertEqual(result.current, 2.1)
            self.assertEqual(result.power, 26.25)
            self.assertEqual(result.temperature, 25.0)
            self.assertEqual(result.irradiance, 800.0)
            
            # Assert that the storage.store method was called with the correct data
            self.mock_storage.store.assert_called_once()
            stored_data = self.mock_storage.store.call_args[0][0]
            self.assertIsInstance(stored_data, SolarData)
            self.assertEqual(stored_data.voltage, 12.5)
            self.assertEqual(stored_data.current, 2.1)
            self.assertEqual(stored_data.power, 26.25)
            self.assertEqual(stored_data.temperature, 25.0)
            self.assertEqual(stored_data.irradiance, 800.0)


class TestInfluxDBStorage(unittest.TestCase):
    """Tests for the InfluxDBStorage class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the InfluxDBClient
        self.mock_client = MagicMock()
        self.storage = InfluxDBStorage(
            client=self.mock_client,
            database="solar_data",
            measurement="solar_metrics"
        )
    
    def test_store(self):
        """Test storing data in InfluxDB."""
        # Create a SolarData object
        data = SolarData(
            timestamp=datetime.now(),
            voltage=12.5,
            current=2.1,
            power=26.25,
            temperature=25.0,
            irradiance=800.0,
            metadata={"panel_id": "panel1"}
        )
        
        # Call the store method
        self.storage.store(data)
        
        # Assert that the client.write_points method was called with the correct data
        self.mock_client.write_points.assert_called_once()
        points = self.mock_client.write_points.call_args[0][0]
        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual(point["measurement"], "solar_metrics")
        self.assertEqual(point["tags"]["panel_id"], "panel1")
        self.assertEqual(point["fields"]["voltage"], 12.5)
        self.assertEqual(point["fields"]["current"], 2.1)
        self.assertEqual(point["fields"]["power"], 26.25)
        self.assertEqual(point["fields"]["temperature"], 25.0)
        self.assertEqual(point["fields"]["irradiance"], 800.0)
    
    def test_query(self):
        """Test querying data from InfluxDB."""
        # Mock the query result
        mock_result = {
            "results": [
                {
                    "series": [
                        {
                            "name": "solar_metrics",
                            "columns": ["time", "voltage", "current", "power", "temperature", "irradiance", "panel_id"],
                            "values": [
                                ["2025-04-22T12:00:00Z", 12.5, 2.1, 26.25, 25.0, 800.0, "panel1"],
                                ["2025-04-22T12:01:00Z", 12.6, 2.2, 27.72, 25.1, 810.0, "panel1"]
                            ]
                        }
                    ]
                }
            ]
        }
        self.mock_client.query.return_value = mock_result
        
        # Call the query method
        start_time = "2025-04-22T12:00:00Z"
        end_time = "2025-04-22T12:01:00Z"
        panel_id = "panel1"
        result = self.storage.query(start_time, end_time, panel_id)
        
        # Assert that the client.query method was called with the correct query
        self.mock_client.query.assert_called_once()
        query = self.mock_client.query.call_args[0][0]
        self.assertIn("SELECT * FROM solar_metrics", query)
        self.assertIn(f"panel_id = '{panel_id}'", query)
        self.assertIn(f"time >= '{start_time}'", query)
        self.assertIn(f"time <= '{end_time}'", query)
        
        # Assert that the result is a list of SolarData objects with the correct values
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], SolarData)
        self.assertEqual(result[0].voltage, 12.5)
        self.assertEqual(result[0].current, 2.1)
        self.assertEqual(result[0].power, 26.25)
        self.assertEqual(result[0].temperature, 25.0)
        self.assertEqual(result[0].irradiance, 800.0)
        self.assertEqual(result[0].metadata["panel_id"], "panel1")
        
        self.assertIsInstance(result[1], SolarData)
        self.assertEqual(result[1].voltage, 12.6)
        self.assertEqual(result[1].current, 2.2)
        self.assertEqual(result[1].power, 27.72)
        self.assertEqual(result[1].temperature, 25.1)
        self.assertEqual(result[1].irradiance, 810.0)
        self.assertEqual(result[1].metadata["panel_id"], "panel1")


if __name__ == "__main__":
    unittest.main()
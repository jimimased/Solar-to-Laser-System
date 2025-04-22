"""
Storage implementation for solar data.

This module provides classes for storing solar data in databases.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from ..common import SolarData

logger = logging.getLogger(__name__)


class StorageInterface:
    """Base class for storage interfaces."""
    
    def store(self, data: SolarData) -> bool:
        """Store solar data.
        
        Args:
            data: Solar data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement store()")
    
    def retrieve(
        self,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> List[SolarData]:
        """Retrieve solar data for a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            fields: Fields to retrieve (None for all)
        
        Returns:
            List[SolarData]: Retrieved data
        """
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def get_statistics(
        self,
        field: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """Get statistics for a field over a time range.
        
        Args:
            field: Field to get statistics for
            start_time: Start time
            end_time: End time
            aggregation: Aggregation function (mean, min, max, sum)
        
        Returns:
            Dict[str, float]: Statistics
        """
        raise NotImplementedError("Subclasses must implement get_statistics()")


class InfluxDBStorage(StorageInterface):
    """Storage interface for InfluxDB."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8086,
        database: str = "solar_data",
        username: Optional[str] = None,
        password: Optional[str] = None,
        retention_policy: str = "autogen"
    ):
        """Initialize the InfluxDB storage interface.
        
        Args:
            host: InfluxDB host
            port: InfluxDB port
            database: InfluxDB database
            username: InfluxDB username
            password: InfluxDB password
            retention_policy: InfluxDB retention policy
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.retention_policy = retention_policy
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to InfluxDB."""
        try:
            from influxdb import InfluxDBClient
            
            self.client = InfluxDBClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database
            )
            
            # Create database if it doesn't exist
            databases = self.client.get_list_database()
            if {"name": self.database} not in databases:
                self.client.create_database(self.database)
                logger.info(f"Created database {self.database}")
            
            # Switch to the database
            self.client.switch_database(self.database)
            
            logger.info(f"Connected to InfluxDB at {self.host}:{self.port}")
        except ImportError:
            logger.error("influxdb not installed. Install with 'pip install influxdb'")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise
    
    def store(self, data: SolarData) -> bool:
        """Store solar data in InfluxDB.
        
        Args:
            data: Solar data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            self._connect()
        
        try:
            # Convert SolarData to InfluxDB point format
            point = {
                "measurement": "solar_metrics",
                "tags": {
                    "source": data.metadata.get("source", "default")
                },
                "time": data.timestamp.isoformat(),
                "fields": {
                    "voltage": float(data.voltage),
                    "current": float(data.current),
                    "power": float(data.power)
                }
            }
            
            # Add optional fields if present
            if data.temperature is not None:
                point["fields"]["temperature"] = float(data.temperature)
            
            if data.irradiance is not None:
                point["fields"]["irradiance"] = float(data.irradiance)
            
            # Add weather data if present
            weather = data.metadata.get("weather")
            if weather:
                for key, value in weather.items():
                    if isinstance(value, (int, float)):
                        point["fields"][f"weather_{key}"] = float(value)
            
            # Write to InfluxDB
            self.client.write_points([point], retention_policy=self.retention_policy)
            
            return True
        except Exception as e:
            logger.error(f"Error storing data in InfluxDB: {e}")
            return False
    
    def retrieve(
        self,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> List[SolarData]:
        """Retrieve solar data from InfluxDB for a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            fields: Fields to retrieve (None for all)
        
        Returns:
            List[SolarData]: Retrieved data
        """
        if not self.client:
            self._connect()
        
        try:
            # Build query
            field_str = "*" if not fields else ", ".join(fields)
            query = f"""
                SELECT {field_str}
                FROM solar_metrics
                WHERE time >= '{start_time.isoformat()}'
                AND time <= '{end_time.isoformat()}'
            """
            
            # Execute query
            result = self.client.query(query)
            
            # Convert result to SolarData objects
            solar_data_list = []
            for point in result.get_points(measurement="solar_metrics"):
                # Extract timestamp
                timestamp = datetime.fromisoformat(point["time"].replace("Z", "+00:00"))
                
                # Extract fields
                voltage = point.get("voltage", 0.0)
                current = point.get("current", 0.0)
                power = point.get("power", 0.0)
                temperature = point.get("temperature")
                irradiance = point.get("irradiance")
                
                # Extract weather data
                metadata = {}
                weather = {}
                for key, value in point.items():
                    if key.startswith("weather_"):
                        weather_key = key[8:]  # Remove "weather_" prefix
                        weather[weather_key] = value
                
                if weather:
                    metadata["weather"] = weather
                
                # Create SolarData object
                solar_data = SolarData(
                    timestamp=timestamp,
                    voltage=voltage,
                    current=current,
                    power=power,
                    temperature=temperature,
                    irradiance=irradiance,
                    metadata=metadata
                )
                
                solar_data_list.append(solar_data)
            
            return solar_data_list
        except Exception as e:
            logger.error(f"Error retrieving data from InfluxDB: {e}")
            return []
    
    def get_statistics(
        self,
        field: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """Get statistics for a field over a time range from InfluxDB.
        
        Args:
            field: Field to get statistics for
            start_time: Start time
            end_time: End time
            aggregation: Aggregation function (mean, min, max, sum)
        
        Returns:
            Dict[str, float]: Statistics
        """
        if not self.client:
            self._connect()
        
        try:
            # Validate aggregation function
            valid_aggregations = ["mean", "min", "max", "sum", "count"]
            if aggregation not in valid_aggregations:
                raise ValueError(f"Invalid aggregation function: {aggregation}. Must be one of {valid_aggregations}")
            
            # Build query
            query = f"""
                SELECT {aggregation}("{field}") AS value
                FROM solar_metrics
                WHERE time >= '{start_time.isoformat()}'
                AND time <= '{end_time.isoformat()}'
            """
            
            # Execute query
            result = self.client.query(query)
            
            # Extract result
            points = list(result.get_points())
            if not points:
                return {aggregation: 0.0}
            
            return {aggregation: points[0]["value"]}
        except Exception as e:
            logger.error(f"Error getting statistics from InfluxDB: {e}")
            return {aggregation: 0.0}
    
    def get_time_series(
        self,
        field: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> List[Tuple[datetime, float]]:
        """Get time series data for a field over a time range from InfluxDB.
        
        Args:
            field: Field to get time series for
            start_time: Start time
            end_time: End time
            interval: Time interval for grouping (e.g., 1h, 5m, 30s)
        
        Returns:
            List[Tuple[datetime, float]]: Time series data
        """
        if not self.client:
            self._connect()
        
        try:
            # Build query
            query = f"""
                SELECT mean("{field}") AS value
                FROM solar_metrics
                WHERE time >= '{start_time.isoformat()}'
                AND time <= '{end_time.isoformat()}'
                GROUP BY time({interval})
            """
            
            # Execute query
            result = self.client.query(query)
            
            # Extract result
            time_series = []
            for point in result.get_points():
                timestamp = datetime.fromisoformat(point["time"].replace("Z", "+00:00"))
                value = point["value"]
                if value is not None:
                    time_series.append((timestamp, value))
            
            return time_series
        except Exception as e:
            logger.error(f"Error getting time series from InfluxDB: {e}")
            return []
    
    def create_continuous_query(
        self,
        name: str,
        field: str,
        aggregation: str,
        interval: str = "1h",
        retention_policy: Optional[str] = None
    ) -> bool:
        """Create a continuous query for downsampling data.
        
        Args:
            name: Name of the continuous query
            field: Field to aggregate
            aggregation: Aggregation function (mean, min, max, sum)
            interval: Time interval for grouping (e.g., 1h, 5m, 30s)
            retention_policy: Retention policy for the downsampled data
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            self._connect()
        
        try:
            # Validate aggregation function
            valid_aggregations = ["mean", "min", "max", "sum", "count"]
            if aggregation not in valid_aggregations:
                raise ValueError(f"Invalid aggregation function: {aggregation}. Must be one of {valid_aggregations}")
            
            # Set default retention policy if not provided
            rp = retention_policy or self.retention_policy
            
            # Build query
            query = f"""
                CREATE CONTINUOUS QUERY "{name}" ON "{self.database}"
                BEGIN
                    SELECT {aggregation}("{field}") AS "{field}"
                    INTO "{rp}"."downsampled_{field}_{interval}"
                    FROM solar_metrics
                    GROUP BY time({interval})
                END
            """
            
            # Execute query
            self.client.query(query)
            
            logger.info(f"Created continuous query {name}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating continuous query: {e}")
            return False
    
    def create_retention_policy(
        self,
        name: str,
        duration: str,
        replication: int = 1,
        default: bool = False
    ) -> bool:
        """Create a retention policy.
        
        Args:
            name: Name of the retention policy
            duration: Duration to keep data (e.g., 30d, 6w, 1y)
            replication: Replication factor
            default: Whether to make this the default retention policy
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            self._connect()
        
        try:
            # Build query
            default_str = "DEFAULT" if default else ""
            query = f"""
                CREATE RETENTION POLICY "{name}"
                ON "{self.database}"
                DURATION {duration}
                REPLICATION {replication}
                {default_str}
            """
            
            # Execute query
            self.client.query(query)
            
            logger.info(f"Created retention policy {name}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating retention policy: {e}")
            return False


class PostgreSQLStorage(StorageInterface):
    """Storage interface for PostgreSQL."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "solar_data",
        username: str = "postgres",
        password: str = "postgres",
        table: str = "solar_data"
    ):
        """Initialize the PostgreSQL storage interface.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: PostgreSQL database
            username: PostgreSQL username
            password: PostgreSQL password
            table: PostgreSQL table
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.table = table
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            import psycopg2.extras
            
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            
            # Create table if it doesn't exist
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        voltage FLOAT NOT NULL,
                        current FLOAT NOT NULL,
                        power FLOAT NOT NULL,
                        temperature FLOAT,
                        irradiance FLOAT,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create index on timestamp
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table}_timestamp
                    ON {self.table}(timestamp)
                """)
                
                self.conn.commit()
            
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}")
        except ImportError:
            logger.error("psycopg2 not installed. Install with 'pip install psycopg2-binary'")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def store(self, data: SolarData) -> bool:
        """Store solar data in PostgreSQL.
        
        Args:
            data: Solar data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.conn:
            self._connect()
        
        try:
            import json
            
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table}
                    (timestamp, voltage, current, power, temperature, irradiance, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    data.timestamp,
                    data.voltage,
                    data.current,
                    data.power,
                    data.temperature,
                    data.irradiance,
                    json.dumps(data.metadata)
                ))
                
                self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Error storing data in PostgreSQL: {e}")
            self.conn.rollback()
            return False
    
    def retrieve(
        self,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> List[SolarData]:
        """Retrieve solar data from PostgreSQL for a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            fields: Fields to retrieve (None for all)
        
        Returns:
            List[SolarData]: Retrieved data
        """
        if not self.conn:
            self._connect()
        
        try:
            # Build query
            field_str = "*" if not fields else ", ".join(fields)
            
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT {field_str}
                    FROM {self.table}
                    WHERE timestamp >= %s
                    AND timestamp <= %s
                    ORDER BY timestamp
                """, (start_time, end_time))
                
                rows = cur.fetchall()
            
            # Convert rows to SolarData objects
            solar_data_list = []
            for row in rows:
                # Extract fields based on query result
                if fields:
                    # Custom fields query
                    data = dict(zip(fields, row))
                    solar_data = SolarData(
                        timestamp=data["timestamp"],
                        voltage=data.get("voltage", 0.0),
                        current=data.get("current", 0.0),
                        power=data.get("power", 0.0),
                        temperature=data.get("temperature"),
                        irradiance=data.get("irradiance"),
                        metadata=data.get("metadata", {})
                    )
                else:
                    # All fields query
                    solar_data = SolarData(
                        timestamp=row[1],  # timestamp
                        voltage=row[2],    # voltage
                        current=row[3],    # current
                        power=row[4],      # power
                        temperature=row[5],  # temperature
                        irradiance=row[6],   # irradiance
                        metadata=row[7]      # metadata
                    )
                
                solar_data_list.append(solar_data)
            
            return solar_data_list
        except Exception as e:
            logger.error(f"Error retrieving data from PostgreSQL: {e}")
            return []
    
    def get_statistics(
        self,
        field: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """Get statistics for a field over a time range from PostgreSQL.
        
        Args:
            field: Field to get statistics for
            start_time: Start time
            end_time: End time
            aggregation: Aggregation function (mean, min, max, sum)
        
        Returns:
            Dict[str, float]: Statistics
        """
        if not self.conn:
            self._connect()
        
        try:
            # Map aggregation function to SQL
            agg_map = {
                "mean": "AVG",
                "min": "MIN",
                "max": "MAX",
                "sum": "SUM",
                "count": "COUNT"
            }
            
            if aggregation not in agg_map:
                raise ValueError(f"Invalid aggregation function: {aggregation}. Must be one of {list(agg_map.keys())}")
            
            sql_agg = agg_map[aggregation]
            
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT {sql_agg}({field}) AS value
                    FROM {self.table}
                    WHERE timestamp >= %s
                    AND timestamp <= %s
                """, (start_time, end_time))
                
                result = cur.fetchone()
            
            return {aggregation: result[0] if result[0] is not None else 0.0}
        except Exception as e:
            logger.error(f"Error getting statistics from PostgreSQL: {e}")
            return {aggregation: 0.0}


class FileStorage(StorageInterface):
    """Storage interface for file-based storage."""
    
    def __init__(self, file_path: str, format: str = "csv"):
        """Initialize the file storage interface.
        
        Args:
            file_path: Path to the file
            format: File format (csv, json)
        """
        self.file_path = file_path
        self.format = format.lower()
        
        # Validate format
        if self.format not in ["csv", "json"]:
            raise ValueError(f"Invalid format: {format}. Must be one of ['csv', 'json']")
        
        # Create file if it doesn't exist
        self._create_file()
    
    def _create_file(self):
        """Create the file if it doesn't exist."""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Create file if it doesn't exist
        if not os.path.exists(self.file_path):
            if self.format == "csv":
                with open(self.file_path, "w") as f:
                    f.write("timestamp,voltage,current,power,temperature,irradiance,metadata\n")
            elif self.format == "json":
                with open(self.file_path, "w") as f:
                    f.write("[]")
    
    def store(self, data: SolarData) -> bool:
        """Store solar data in a file.
        
        Args:
            data: Solar data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.format == "csv":
                return self._store_csv(data)
            elif self.format == "json":
                return self._store_json(data)
            else:
                return False
        except Exception as e:
            logger.error(f"Error storing data in file: {e}")
            return False
    
    def _store_csv(self, data: SolarData) -> bool:
        """Store solar data in a CSV file.
        
        Args:
            data: Solar data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        import csv
        
        try:
            with open(self.file_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    data.timestamp.isoformat(),
                    data.voltage,
                    data.current,
                    data.power,
                    data.temperature if data.temperature is not None else "",
                    data.irradiance if data.irradiance is not None else "",
                    str(data.metadata) if data.metadata else ""
                ])
            
            return True
        except Exception as e:
            logger.error(f"Error storing data in CSV file: {e}")
            return False
    
    def _store_json(self, data: SolarData) -> bool:
        """Store solar data in a JSON file.
        
        Args:
            data: Solar data to store
        
        Returns:
            bool: True if successful, False otherwise
        """
        import json
        
        try:
            # Read existing data
            with open(self.file_path, "r") as f:
                existing_data = json.load(f)
            
            # Append new data
            existing_data.append(data.to_dict())
            
            # Write back to file
            with open(self.file_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error storing data in JSON file: {e}")
            return False
    
    def retrieve(
        self,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> List[SolarData]:
        """Retrieve solar data from a file for a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            fields: Fields to retrieve (None for all)
        
        Returns:
            List[SolarData]: Retrieved data
        """
        try:
            if self.format == "csv":
                return self._retrieve_csv(start_time, end_time, fields)
            elif self.format == "json":
                return self._retrieve_json(start_time, end_time, fields)
            else:
                return []
        except Exception as e:
            logger.error(f"Error retrieving data from file: {e}")
            return []
    
    def _retrieve_csv(
        self,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> List[SolarData]:
        """Retrieve solar data from a CSV file for a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            fields: Fields to retrieve (None for all)
        
        Returns:
            List[SolarData]: Retrieved data
        """
        import csv
        
        try:
            solar_data_list = []
            
            with open(self.file_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(row[0])
                    
                    # Check if timestamp is in range
                    if start_time <= timestamp <= end_time:
                        # Parse fields
                        voltage = float(row[1])
                        current = float(row[2])
                        power = float(row[3])
                        temperature = float(row[4]) if row[4] else None
                        irradiance = float(row[5]) if row[5] else None
                        metadata = eval(row[6]) if row[6] else {}
                        
                        # Create SolarData object
                        solar_data = SolarData(
                            timestamp=timestamp,
                            voltage=voltage,
                            current=current,
                            power=power,
                            temperature=temperature,
                            irradiance=irradiance,
                            metadata=metadata
                        )
                        
                        solar_data_list.append(solar_data)
            
            return solar_data_list
        except Exception as e:
            logger.error(f"Error retrieving data from CSV file: {e}")
            return []
    
    def _retrieve_json(
        self,
        start_time: datetime,
        end_time: datetime,
        fields: Optional[List[str]] = None
    ) -> List[SolarData]:
        """Retrieve solar data from a JSON file for a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            fields: Fields to retrieve (None for all)
        
        Returns:
            List[SolarData]: Retrieved data
        """
        import json
        
        try:
            # Read data from file
            with open(self.file_path, "r") as f:
                data = json.load(f)
            
            # Filter data by time range
            solar_data_list = []
            for item in data:
                # Parse timestamp
                timestamp = datetime.fromisoformat(item["timestamp"])
                
                # Check if timestamp is in range
                if start_time <= timestamp <= end_time:
                    # Create SolarData object
                    solar_data = SolarData.from_dict(item)
                    solar_data_list.append(solar_data)
            
            return solar_data_list
        except Exception as e:
            logger.error(f"Error retrieving data from JSON file: {e}")
            return []
    
    def get_statistics(
        self,
        field: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """Get statistics for a field over a time range from a file.
        
        Args:
            field: Field to get statistics for
            start_time: Start time
            end_time: End time
            aggregation: Aggregation function (mean, min, max, sum)
        
        Returns:
            Dict[str, float]: Statistics
        """
        try:
            # Retrieve data for the time range
            data = self.retrieve(start_time, end_time)
            
            # Extract field values
            values = []
            for item in data:
                value = getattr(item, field, None)
                if value is not None:
                    values.append(float(value))
            
            # Calculate statistics
            if not values:
                return {aggregation: 0.0}
            
            if aggregation == "mean":
                result = sum(values) / len(values)
            elif aggregation == "min":
                result = min(values)
            elif aggregation == "max":
                result = max(values)
            elif aggregation == "sum":
                result = sum(values)
            elif aggregation == "count":
                result = len(values)
            else:
                raise ValueError(f"Invalid aggregation function: {aggregation}")
            
            return {aggregation: result}
        except Exception as e:
            logger.error(f"Error getting statistics from file: {e}")
            return {aggregation: 0.0}
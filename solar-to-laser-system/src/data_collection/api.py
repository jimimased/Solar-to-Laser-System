"""
API endpoints for the data collection module.

This module provides FastAPI endpoints for accessing solar data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..common import SolarData
from .collector import SolarDataCollector, SimulatedSensorInterface, WeatherDataProvider
from .storage import StorageInterface, InfluxDBStorage

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses

class SolarDataModel(BaseModel):
    """Pydantic model for solar data."""
    
    timestamp: str
    voltage: float
    current: float
    power: float
    temperature: Optional[float] = None
    irradiance: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        
        schema_extra = {
            "example": {
                "timestamp": "2025-04-22T12:00:00",
                "voltage": 24.5,
                "current": 3.2,
                "power": 78.4,
                "temperature": 25.3,
                "irradiance": 850.0,
                "metadata": {
                    "source": "panel_1",
                    "weather": {
                        "temperature": 28.5,
                        "humidity": 65.0,
                        "cloud_cover": 10.0
                    }
                }
            }
        }
    
    @classmethod
    def from_solar_data(cls, solar_data: SolarData) -> "SolarDataModel":
        """Create a SolarDataModel from a SolarData object."""
        return cls(
            timestamp=solar_data.timestamp.isoformat(),
            voltage=solar_data.voltage,
            current=solar_data.current,
            power=solar_data.power,
            temperature=solar_data.temperature,
            irradiance=solar_data.irradiance,
            metadata=solar_data.metadata
        )
    
    def to_solar_data(self) -> SolarData:
        """Convert to a SolarData object."""
        return SolarData(
            timestamp=datetime.fromisoformat(self.timestamp),
            voltage=self.voltage,
            current=self.current,
            power=self.power,
            temperature=self.temperature,
            irradiance=self.irradiance,
            metadata=self.metadata
        )


class SolarDataBatch(BaseModel):
    """Pydantic model for a batch of solar data."""
    
    data: List[SolarDataModel]


class StatisticsResponse(BaseModel):
    """Pydantic model for statistics response."""
    
    field: str
    start_time: str
    end_time: str
    aggregation: str
    value: float


class TimeSeriesResponse(BaseModel):
    """Pydantic model for time series response."""
    
    field: str
    start_time: str
    end_time: str
    interval: str
    data: List[Dict[str, Union[str, float]]]


class StatusResponse(BaseModel):
    """Pydantic model for status response."""
    
    status: str
    message: str
    timestamp: str


# API application

def create_api(
    storage: StorageInterface,
    collector: Optional[SolarDataCollector] = None
) -> FastAPI:
    """Create a FastAPI application for the data collection API.
    
    Args:
        storage: Storage interface
        collector: Solar data collector (optional)
    
    Returns:
        FastAPI: FastAPI application
    """
    app = FastAPI(
        title="Solar Data Collection API",
        description="API for collecting and accessing solar panel data",
        version="1.0.0"
    )
    
    # Dependency for storage
    def get_storage() -> StorageInterface:
        return storage
    
    # Dependency for collector
    def get_collector() -> Optional[SolarDataCollector]:
        return collector
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Solar Data Collection API",
            "version": "1.0.0",
            "documentation": "/docs"
        }
    
    @app.get("/status", response_model=StatusResponse, tags=["General"])
    async def get_status():
        """Get API status."""
        return StatusResponse(
            status="ok",
            message="API is running",
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/api/solar/data", response_model=StatusResponse, tags=["Data"])
    async def post_solar_data(
        data: SolarDataModel,
        storage: StorageInterface = Depends(get_storage)
    ):
        """Store solar data."""
        try:
            # Convert to SolarData
            solar_data = data.to_solar_data()
            
            # Store in database
            success = storage.store(solar_data)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to store data")
            
            return StatusResponse(
                status="success",
                message="Data stored successfully",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/solar/data/batch", response_model=StatusResponse, tags=["Data"])
    async def post_solar_data_batch(
        batch: SolarDataBatch,
        storage: StorageInterface = Depends(get_storage)
    ):
        """Store a batch of solar data."""
        try:
            # Convert to SolarData objects
            solar_data_list = [item.to_solar_data() for item in batch.data]
            
            # Store in database
            success_count = 0
            for solar_data in solar_data_list:
                if storage.store(solar_data):
                    success_count += 1
            
            return StatusResponse(
                status="success",
                message=f"Stored {success_count}/{len(solar_data_list)} data points",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error storing batch data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/solar/data/{start_time}/{end_time}", response_model=List[SolarDataModel], tags=["Data"])
    async def get_solar_data(
        start_time: str,
        end_time: str,
        fields: Optional[str] = None,
        storage: StorageInterface = Depends(get_storage)
    ):
        """Get solar data for a time range."""
        try:
            # Parse time range
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            
            # Parse fields
            field_list = fields.split(",") if fields else None
            
            # Retrieve data
            solar_data_list = storage.retrieve(start, end, field_list)
            
            # Convert to response models
            return [SolarDataModel.from_solar_data(data) for data in solar_data_list]
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/solar/stats/{field}/{period}", response_model=StatisticsResponse, tags=["Statistics"])
    async def get_solar_stats(
        field: str,
        period: str,
        aggregation: str = "mean",
        end_time: Optional[str] = None,
        storage: StorageInterface = Depends(get_storage)
    ):
        """Get statistics for a field over a time period."""
        try:
            # Parse end time (default to now)
            end = datetime.fromisoformat(end_time) if end_time else datetime.now()
            
            # Parse period
            if period.endswith("h"):
                hours = int(period[:-1])
                start = end - timedelta(hours=hours)
            elif period.endswith("d"):
                days = int(period[:-1])
                start = end - timedelta(days=days)
            elif period.endswith("w"):
                weeks = int(period[:-1])
                start = end - timedelta(weeks=weeks)
            else:
                raise HTTPException(status_code=400, detail="Invalid period format")
            
            # Get statistics
            stats = storage.get_statistics(field, start, end, aggregation)
            
            return StatisticsResponse(
                field=field,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
                aggregation=aggregation,
                value=stats.get(aggregation, 0.0)
            )
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/solar/timeseries/{field}/{period}", response_model=TimeSeriesResponse, tags=["Statistics"])
    async def get_solar_timeseries(
        field: str,
        period: str,
        interval: str = "1h",
        end_time: Optional[str] = None,
        storage: StorageInterface = Depends(get_storage)
    ):
        """Get time series data for a field over a time period."""
        try:
            # Check if storage supports time series
            if not hasattr(storage, "get_time_series"):
                raise HTTPException(status_code=400, detail="Storage does not support time series")
            
            # Parse end time (default to now)
            end = datetime.fromisoformat(end_time) if end_time else datetime.now()
            
            # Parse period
            if period.endswith("h"):
                hours = int(period[:-1])
                start = end - timedelta(hours=hours)
            elif period.endswith("d"):
                days = int(period[:-1])
                start = end - timedelta(days=days)
            elif period.endswith("w"):
                weeks = int(period[:-1])
                start = end - timedelta(weeks=weeks)
            else:
                raise HTTPException(status_code=400, detail="Invalid period format")
            
            # Get time series
            time_series = storage.get_time_series(field, start, end, interval)
            
            # Convert to response format
            data = [
                {"timestamp": timestamp.isoformat(), "value": value}
                for timestamp, value in time_series
            ]
            
            return TimeSeriesResponse(
                field=field,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
                interval=interval,
                data=data
            )
        except Exception as e:
            logger.error(f"Error getting time series: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/solar/collect", response_model=StatusResponse, tags=["Collection"])
    async def trigger_collection(
        background_tasks: BackgroundTasks,
        duration: int = 60,
        interval: float = 1.0,
        collector: Optional[SolarDataCollector] = Depends(get_collector),
        storage: StorageInterface = Depends(get_storage)
    ):
        """Trigger data collection for a specified duration."""
        if not collector:
            raise HTTPException(status_code=400, detail="Collector not available")
        
        try:
            # Define collection task
            def collect_data():
                start_time = datetime.now()
                end_time = start_time + timedelta(seconds=duration)
                
                logger.info(f"Starting data collection for {duration} seconds")
                
                while datetime.now() < end_time:
                    # Collect data
                    solar_data = collector.collect_single()
                    
                    # Store data
                    storage.store(solar_data)
                    
                    # Wait for next collection
                    import time
                    time.sleep(interval)
                
                logger.info("Data collection completed")
            
            # Add task to background tasks
            background_tasks.add_task(collect_data)
            
            return StatusResponse(
                status="success",
                message=f"Data collection started for {duration} seconds",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Error triggering collection: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/solar/simulate", response_model=SolarDataModel, tags=["Simulation"])
    async def simulate_data():
        """Generate simulated solar data."""
        try:
            # Create simulated sensor interface
            sensor = SimulatedSensorInterface()
            
            # Create weather provider
            weather_provider = None
            try:
                # Try to create a real weather provider, but fall back to None if it fails
                weather_provider = WeatherDataProvider(
                    api_key="dummy_key",
                    location="London"
                )
            except:
                pass
            
            # Create collector
            collector = SolarDataCollector(
                sensor_interface=sensor,
                weather_provider=weather_provider
            )
            
            # Collect data
            solar_data = collector.collect_single()
            
            # Convert to response model
            return SolarDataModel.from_solar_data(solar_data)
        except Exception as e:
            logger.error(f"Error simulating data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default API instance
default_storage = InfluxDBStorage(
    host="localhost",
    port=8086,
    database="solar_data"
)

default_sensor = SimulatedSensorInterface()
default_collector = SolarDataCollector(sensor_interface=default_sensor)

api = create_api(default_storage, default_collector)
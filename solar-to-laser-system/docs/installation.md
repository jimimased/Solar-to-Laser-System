# Installation Guide

This guide provides instructions for setting up the Solar-to-Laser System in different environments.

## Prerequisites

### Hardware Requirements
- Raspberry Pi 4 (or Arduino) for data collection
- Solar panels with voltage and current sensors
- Laser projection system compatible with ILDA format
- Server with GPU support for RAVE model training (recommended)

### Software Requirements
- Docker and Docker Compose
- Kubernetes (for production deployment)
- Python 3.9+
- Node.js 16+
- Git

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/solar-to-laser-system.git
cd solar-to-laser-system
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Node.js Environment

```bash
# Install Node.js dependencies
cd src/web
npm install
```

### 4. Set Up Databases

```bash
# Start the database containers
docker-compose up -d influxdb postgres
```

### 5. Configure Environment Variables

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Docker Development Environment

For a complete development environment using Docker:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

This will start all required services:
- InfluxDB for time-series data
- PostgreSQL for relational data
- MQTT broker for data transmission
- Node-RED for data flow management
- FastAPI backend services
- Streamlit dashboards
- Web interface

## Production Deployment

For production deployment using Kubernetes:

### 1. Build and Push Docker Images

```bash
# Build images
docker-compose build

# Tag images
docker tag solar-to-laser-system/data-collection:latest your-registry/solar-to-laser-system/data-collection:latest
docker tag solar-to-laser-system/audio-conversion:latest your-registry/solar-to-laser-system/audio-conversion:latest
docker tag solar-to-laser-system/rave-processing:latest your-registry/solar-to-laser-system/rave-processing:latest
docker tag solar-to-laser-system/vector-generation:latest your-registry/solar-to-laser-system/vector-generation:latest
docker tag solar-to-laser-system/laser-control:latest your-registry/solar-to-laser-system/laser-control:latest
docker tag solar-to-laser-system/web:latest your-registry/solar-to-laser-system/web:latest

# Push images to registry
docker push your-registry/solar-to-laser-system/data-collection:latest
docker push your-registry/solar-to-laser-system/audio-conversion:latest
docker push your-registry/solar-to-laser-system/rave-processing:latest
docker push your-registry/solar-to-laser-system/vector-generation:latest
docker push your-registry/solar-to-laser-system/laser-control:latest
docker push your-registry/solar-to-laser-system/web:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/storage.yaml
kubectl apply -f kubernetes/databases.yaml
kubectl apply -f kubernetes/services.yaml
kubectl apply -f kubernetes/deployments.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n solar-to-laser-system

# Check services
kubectl get services -n solar-to-laser-system

# Check ingress
kubectl get ingress -n solar-to-laser-system
```

## Hardware Setup

### Solar Panel Connection

1. Connect the solar panel to the voltage and current sensors
2. Connect the sensors to the Raspberry Pi/Arduino analog inputs
3. Install the data collection software on the Raspberry Pi/Arduino

### Laser System Connection

1. Connect the laser system to a computer running the Laser Control module
2. Configure the laser system according to the manufacturer's instructions
3. Set up the Laser Control module to communicate with the laser system

## Troubleshooting

### Common Issues

#### Database Connection Errors
- Verify that the database containers are running
- Check the database connection settings in the .env file
- Ensure that the database ports are accessible

#### MQTT Connection Issues
- Verify that the MQTT broker is running
- Check the MQTT connection settings in the .env file
- Ensure that the MQTT ports are accessible

#### Laser Control Issues
- Verify that the laser system is connected and powered on
- Check the laser system configuration
- Ensure that the correct drivers are installed

For more detailed troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).
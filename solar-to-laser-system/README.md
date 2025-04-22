# Solar-to-Laser System

A comprehensive system that converts solar panel data into laser imagery through audio processing.

## System Overview

This system follows a pipeline approach:

1. **Data Collection**: Collect data from solar panels (voltage, current, power output)
2. **Audio Conversion**: Convert solar data into audio files
3. **RAVE Processing**: Process audio files using RAVE (Realtime Audio Variational autoEncoder)
4. **Vector Generation**: Extract features from processed audio and convert to vector graphics
5. **Laser Control**: Generate laser control files (ILDA format) from vectors
6. **Global Deployment**: Deploy the system globally through cloud infrastructure

## Architecture

The system is built with a modular architecture, allowing each component to be developed, tested, and deployed independently.

![System Architecture](docs/images/architecture.png)

## Installation

### Prerequisites

- Python 3.9+
- Node.js 16+
- Docker and Docker Compose (for containerized deployment)
- Kubernetes (for cloud deployment)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/solar-to-laser-system.git
cd solar-to-laser-system
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install Node.js dependencies:

```bash
npm install
```

4. Build the web interface:

```bash
npm run build
```

For more detailed installation instructions, see the [Installation Guide](docs/installation.md).

## Usage

### Running with Docker Compose

The easiest way to run the entire system is with Docker Compose:

```bash
docker-compose up
```

This will start all the services, including:
- InfluxDB for time-series data
- PostgreSQL for relational data
- RabbitMQ for message queuing
- All system modules

### Running Individual Services

You can also run individual services:

```bash
# Run data collection API
python -m src.main data

# Run audio conversion API
python -m src.main audio

# Run RAVE processing API
python -m src.main rave

# Run vector generation API
python -m src.main vector

# Run laser control API
python -m src.main laser

# Run all APIs
python -m src.main all
```

### Web Interface

The web interface is available at http://localhost:80 when running with Docker Compose, or at http://localhost:8000 when running the services individually.

## Deployment

### Local Deployment

For local deployment, use Docker Compose:

```bash
docker-compose up -d
```

### Cloud Deployment

For cloud deployment, use Kubernetes:

```bash
# Deploy to Kubernetes
python -m src.main k8s --namespace solar-to-laser-system --registry your-registry

# Check deployment status
kubectl get pods -n solar-to-laser-system
```

## Development

### Project Structure

```
solar-to-laser-system/
├── docs/                  # Documentation
├── src/                   # Source code
│   ├── common/            # Common utilities and data structures
│   ├── data_collection/   # Data collection module
│   ├── audio_conversion/  # Audio conversion module
│   ├── rave_processing/   # RAVE processing module
│   ├── vector_generation/ # Vector generation module
│   ├── laser_control/     # Laser control module
│   ├── deployment/        # Deployment module
│   ├── web/               # Web interface
│   └── main.py            # Main entry point
├── docker/                # Dockerfiles
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
└── webpack.config.js      # Webpack configuration
```

### Running Tests

```bash
# Run Python tests
pytest

# Run JavaScript tests
npm test
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)
- [API Documentation](docs/api.md)
- [User Manual](docs/user-manual.md)
- [Developer Guide](docs/developer-guide.md)
- [Deployment Guide](docs/deployment.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- RAVE (Realtime Audio Variational autoEncoder) - https://github.com/acids-ircam/RAVE
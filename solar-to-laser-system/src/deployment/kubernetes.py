"""
Kubernetes deployment implementation.

This module provides classes for deploying the system to Kubernetes.
"""

import os
import logging
import yaml
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class KubernetesDeployer:
    """Class for deploying to Kubernetes."""
    
    def __init__(
        self,
        namespace: str = "solar-to-laser-system",
        registry: str = "ghcr.io/yourusername",
        config_path: Optional[str] = None
    ):
        """Initialize the Kubernetes deployer.
        
        Args:
            namespace: Kubernetes namespace
            registry: Container registry
            config_path: Path to Kubernetes config
        """
        self.namespace = namespace
        self.registry = registry
        self.config_path = config_path
        
        # Set KUBECONFIG environment variable if config_path is provided
        if config_path:
            os.environ["KUBECONFIG"] = config_path
    
    def create_namespace(self) -> bool:
        """Create Kubernetes namespace.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create namespace YAML
            namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.namespace}
"""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                temp_file.write(namespace_yaml)
                temp_path = temp_file.name
            
            # Apply namespace
            import subprocess
            result = subprocess.run(
                ["kubectl", "apply", "-f", temp_path],
                capture_output=True,
                text=True
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result.returncode != 0:
                logger.error(f"Error creating namespace: {result.stderr}")
                return False
            
            logger.info(f"Created namespace {self.namespace}")
            return True
        except Exception as e:
            logger.error(f"Error creating namespace: {e}")
            return False
    
    def deploy_database(self) -> bool:
        """Deploy database components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create database YAML
            database_yaml = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: influxdb-pvc
  namespace: {self.namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: {self.namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: influxdb
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: influxdb
  template:
    metadata:
      labels:
        app: influxdb
    spec:
      containers:
      - name: influxdb
        image: influxdb:2.0
        ports:
        - containerPort: 8086
        volumeMounts:
        - name: influxdb-storage
          mountPath: /var/lib/influxdb2
        env:
        - name: DOCKER_INFLUXDB_INIT_MODE
          value: "setup"
        - name: DOCKER_INFLUXDB_INIT_USERNAME
          value: "admin"
        - name: DOCKER_INFLUXDB_INIT_PASSWORD
          value: "password"
        - name: DOCKER_INFLUXDB_INIT_ORG
          value: "solar-to-laser"
        - name: DOCKER_INFLUXDB_INIT_BUCKET
          value: "solar_data"
      volumes:
      - name: influxdb-storage
        persistentVolumeClaim:
          claimName: influxdb-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: influxdb
  namespace: {self.namespace}
spec:
  selector:
    app: influxdb
  ports:
  - port: 8086
    targetPort: 8086
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        env:
        - name: POSTGRES_USER
          value: "postgres"
        - name: POSTGRES_PASSWORD
          value: "postgres"
        - name: POSTGRES_DB
          value: "solar_to_laser"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: {self.namespace}
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
"""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                temp_file.write(database_yaml)
                temp_path = temp_file.name
            
            # Apply database components
            import subprocess
            result = subprocess.run(
                ["kubectl", "apply", "-f", temp_path],
                capture_output=True,
                text=True
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result.returncode != 0:
                logger.error(f"Error deploying database: {result.stderr}")
                return False
            
            logger.info("Deployed database components")
            return True
        except Exception as e:
            logger.error(f"Error deploying database: {e}")
            return False
    
    def deploy_message_queue(self) -> bool:
        """Deploy message queue components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create message queue YAML
            mq_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rabbitmq
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: rabbitmq
  template:
    metadata:
      labels:
        app: rabbitmq
    spec:
      containers:
      - name: rabbitmq
        image: rabbitmq:3-management
        ports:
        - containerPort: 5672
        - containerPort: 15672
---
apiVersion: v1
kind: Service
metadata:
  name: rabbitmq
  namespace: {self.namespace}
spec:
  selector:
    app: rabbitmq
  ports:
  - name: amqp
    port: 5672
    targetPort: 5672
  - name: management
    port: 15672
    targetPort: 15672
"""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                temp_file.write(mq_yaml)
                temp_path = temp_file.name
            
            # Apply message queue components
            import subprocess
            result = subprocess.run(
                ["kubectl", "apply", "-f", temp_path],
                capture_output=True,
                text=True
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result.returncode != 0:
                logger.error(f"Error deploying message queue: {result.stderr}")
                return False
            
            logger.info("Deployed message queue components")
            return True
        except Exception as e:
            logger.error(f"Error deploying message queue: {e}")
            return False
    
    def deploy_services(self) -> bool:
        """Deploy service components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create services YAML
            services_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-collection
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: data-collection
  template:
    metadata:
      labels:
        app: data-collection
    spec:
      containers:
      - name: data-collection
        image: {self.registry}/solar-to-laser-system/data-collection:latest
        ports:
        - containerPort: 8000
        env:
        - name: INFLUXDB_HOST
          value: "influxdb"
        - name: INFLUXDB_PORT
          value: "8086"
        - name: INFLUXDB_DATABASE
          value: "solar_data"
        - name: RABBITMQ_HOST
          value: "rabbitmq"
---
apiVersion: v1
kind: Service
metadata:
  name: data-collection
  namespace: {self.namespace}
spec:
  selector:
    app: data-collection
  ports:
  - port: 8000
    targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audio-conversion
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: audio-conversion
  template:
    metadata:
      labels:
        app: audio-conversion
    spec:
      containers:
      - name: audio-conversion
        image: {self.registry}/solar-to-laser-system/audio-conversion:latest
        ports:
        - containerPort: 8001
        env:
        - name: RABBITMQ_HOST
          value: "rabbitmq"
        - name: POSTGRES_HOST
          value: "postgres"
---
apiVersion: v1
kind: Service
metadata:
  name: audio-conversion
  namespace: {self.namespace}
spec:
  selector:
    app: audio-conversion
  ports:
  - port: 8001
    targetPort: 8001
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rave-processing
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: rave-processing
  template:
    metadata:
      labels:
        app: rave-processing
    spec:
      containers:
      - name: rave-processing
        image: {self.registry}/solar-to-laser-system/rave-processing:latest
        ports:
        - containerPort: 8002
        env:
        - name: RABBITMQ_HOST
          value: "rabbitmq"
        - name: POSTGRES_HOST
          value: "postgres"
---
apiVersion: v1
kind: Service
metadata:
  name: rave-processing
  namespace: {self.namespace}
spec:
  selector:
    app: rave-processing
  ports:
  - port: 8002
    targetPort: 8002
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vector-generation
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: vector-generation
  template:
    metadata:
      labels:
        app: vector-generation
    spec:
      containers:
      - name: vector-generation
        image: {self.registry}/solar-to-laser-system/vector-generation:latest
        ports:
        - containerPort: 8003
        env:
        - name: RABBITMQ_HOST
          value: "rabbitmq"
        - name: POSTGRES_HOST
          value: "postgres"
---
apiVersion: v1
kind: Service
metadata:
  name: vector-generation
  namespace: {self.namespace}
spec:
  selector:
    app: vector-generation
  ports:
  - port: 8003
    targetPort: 8003
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: laser-control
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: laser-control
  template:
    metadata:
      labels:
        app: laser-control
    spec:
      containers:
      - name: laser-control
        image: {self.registry}/solar-to-laser-system/laser-control:latest
        ports:
        - containerPort: 8004
        env:
        - name: RABBITMQ_HOST
          value: "rabbitmq"
        - name: POSTGRES_HOST
          value: "postgres"
---
apiVersion: v1
kind: Service
metadata:
  name: laser-control
  namespace: {self.namespace}
spec:
  selector:
    app: laser-control
  ports:
  - port: 8004
    targetPort: 8004
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: {self.namespace}
spec:
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: {self.registry}/solar-to-laser-system/web:latest
        ports:
        - containerPort: 80
        env:
        - name: DATA_COLLECTION_URL
          value: "http://data-collection:8000"
        - name: AUDIO_CONVERSION_URL
          value: "http://audio-conversion:8001"
        - name: RAVE_PROCESSING_URL
          value: "http://rave-processing:8002"
        - name: VECTOR_GENERATION_URL
          value: "http://vector-generation:8003"
        - name: LASER_CONTROL_URL
          value: "http://laser-control:8004"
---
apiVersion: v1
kind: Service
metadata:
  name: web
  namespace: {self.namespace}
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
"""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                temp_file.write(services_yaml)
                temp_path = temp_file.name
            
            # Apply service components
            import subprocess
            result = subprocess.run(
                ["kubectl", "apply", "-f", temp_path],
                capture_output=True,
                text=True
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result.returncode != 0:
                logger.error(f"Error deploying services: {result.stderr}")
                return False
            
            logger.info("Deployed service components")
            return True
        except Exception as e:
            logger.error(f"Error deploying services: {e}")
            return False
    
    def deploy_ingress(self) -> bool:
        """Deploy ingress components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create ingress YAML
            ingress_yaml = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: solar-to-laser-ingress
  namespace: {self.namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - http:
      paths:
      - path: /api/solar
        pathType: Prefix
        backend:
          service:
            name: data-collection
            port:
              number: 8000
      - path: /api/audio
        pathType: Prefix
        backend:
          service:
            name: audio-conversion
            port:
              number: 8001
      - path: /api/rave
        pathType: Prefix
        backend:
          service:
            name: rave-processing
            port:
              number: 8002
      - path: /api/vector
        pathType: Prefix
        backend:
          service:
            name: vector-generation
            port:
              number: 8003
      - path: /api/laser
        pathType: Prefix
        backend:
          service:
            name: laser-control
            port:
              number: 8004
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web
            port:
              number: 80
"""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                temp_file.write(ingress_yaml)
                temp_path = temp_file.name
            
            # Apply ingress components
            import subprocess
            result = subprocess.run(
                ["kubectl", "apply", "-f", temp_path],
                capture_output=True,
                text=True
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result.returncode != 0:
                logger.error(f"Error deploying ingress: {result.stderr}")
                return False
            
            logger.info("Deployed ingress components")
            return True
        except Exception as e:
            logger.error(f"Error deploying ingress: {e}")
            return False
    
    def deploy_all(self) -> bool:
        """Deploy all components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create namespace
            if not self.create_namespace():
                return False
            
            # Deploy database
            if not self.deploy_database():
                return False
            
            # Deploy message queue
            if not self.deploy_message_queue():
                return False
            
            # Deploy services
            if not self.deploy_services():
                return False
            
            # Deploy ingress
            if not self.deploy_ingress():
                return False
            
            logger.info("Deployed all components")
            return True
        except Exception as e:
            logger.error(f"Error deploying all components: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status.
        
        Returns:
            Dict[str, Any]: Deployment status
        """
        try:
            # Get pod status
            import subprocess
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error getting pod status: {result.stderr}")
                return {"status": "error", "message": result.stderr}
            
            # Parse pod status
            import json
            pods = json.loads(result.stdout)
            
            # Get service status
            result = subprocess.run(
                ["kubectl", "get", "services", "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error getting service status: {result.stderr}")
                return {"status": "error", "message": result.stderr}
            
            # Parse service status
            services = json.loads(result.stdout)
            
            return {
                "status": "ok",
                "pods": pods,
                "services": services
            }
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {"status": "error", "message": str(e)}
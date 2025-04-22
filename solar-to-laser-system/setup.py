"""
Setup script for the Solar-to-Laser System.
"""

from setuptools import setup, find_packages

setup(
    name="solar-to-laser-system",
    version="1.0.0",
    description="A system that converts solar panel data into laser imagery through audio processing",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/solar-to-laser-system",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Common dependencies
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pydantic>=1.8.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        
        # Data collection dependencies
        "influxdb>=5.3.0",
        "paho-mqtt>=1.5.0",
        "requests>=2.26.0",
        "psycopg2-binary>=2.9.0",
        
        # Audio conversion dependencies
        "librosa>=0.8.0",
        "soundfile>=0.10.0",
        
        # RAVE processing dependencies
        "torch>=1.9.0",
        "tqdm>=4.62.0",
        
        # Vector generation dependencies
        "scikit-learn>=0.24.0",
        "svg.path>=4.0.0",
        
        # Laser control dependencies
        "opencv-python>=4.5.0",
        
        # Deployment dependencies
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "solar-to-laser=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
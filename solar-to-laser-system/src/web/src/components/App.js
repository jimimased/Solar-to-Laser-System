import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import io from 'socket.io-client';

// Initialize socket connection
const socket = io();

// Main App component
function App() {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);

  // Connect to socket on component mount
  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to server');
      setConnected(true);
      setError(null);
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
    });

    socket.on('error', (data) => {
      console.error('Socket error:', data.message);
      setError(data.message);
    });

    // Clean up on unmount
    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('error');
    };
  }, []);

  return (
    <Router>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="container">
            <div className="d-flex justify-content-between align-items-center">
              <h1>Solar-to-Laser System</h1>
              <div className="connection-status">
                {connected ? (
                  <span className="badge bg-success">Connected</span>
                ) : (
                  <span className="badge bg-danger">Disconnected</span>
                )}
              </div>
            </div>
            <nav className="mt-2">
              <ul className="nav">
                <li className="nav-item">
                  <Link className="nav-link" to="/">Dashboard</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/data-collection">Data Collection</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/audio-conversion">Audio Conversion</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/rave-processing">RAVE Processing</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/vector-generation">Vector Generation</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/laser-control">Laser Control</Link>
                </li>
              </ul>
            </nav>
          </div>
        </header>

        {/* Main content */}
        <main className="main-content">
          <div className="container">
            {error && (
              <div className="alert alert-danger" role="alert">
                {error}
              </div>
            )}

            <Routes>
              <Route path="/" element={<Dashboard socket={socket} />} />
              <Route path="/data-collection" element={<DataCollection socket={socket} />} />
              <Route path="/audio-conversion" element={<AudioConversion socket={socket} />} />
              <Route path="/rave-processing" element={<RaveProcessing socket={socket} />} />
              <Route path="/vector-generation" element={<VectorGeneration socket={socket} />} />
              <Route path="/laser-control" element={<LaserControl socket={socket} />} />
            </Routes>
          </div>
        </main>

        {/* Footer */}
        <footer className="footer">
          <div className="container">
            <p className="text-center">Solar-to-Laser System &copy; 2025</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

// Dashboard component
function Dashboard({ socket }) {
  return (
    <div className="module-section">
      <h2>Dashboard</h2>
      <p>Welcome to the Solar-to-Laser System dashboard. Use the navigation above to access different modules.</p>
      
      <div className="row mt-4">
        <div className="col-md-4">
          <div className="card">
            <div className="card-header">Data Collection</div>
            <div className="card-body">
              <p>Collect data from solar panels.</p>
              <Link to="/data-collection" className="btn btn-primary">Go to Data Collection</Link>
            </div>
          </div>
        </div>
        
        <div className="col-md-4">
          <div className="card">
            <div className="card-header">Audio Conversion</div>
            <div className="card-body">
              <p>Convert solar data to audio.</p>
              <Link to="/audio-conversion" className="btn btn-primary">Go to Audio Conversion</Link>
            </div>
          </div>
        </div>
        
        <div className="col-md-4">
          <div className="card">
            <div className="card-header">RAVE Processing</div>
            <div className="card-body">
              <p>Process audio using RAVE.</p>
              <Link to="/rave-processing" className="btn btn-primary">Go to RAVE Processing</Link>
            </div>
          </div>
        </div>
      </div>
      
      <div className="row mt-3">
        <div className="col-md-4">
          <div className="card">
            <div className="card-header">Vector Generation</div>
            <div className="card-body">
              <p>Generate vector graphics from processed audio.</p>
              <Link to="/vector-generation" className="btn btn-primary">Go to Vector Generation</Link>
            </div>
          </div>
        </div>
        
        <div className="col-md-4">
          <div className="card">
            <div className="card-header">Laser Control</div>
            <div className="card-body">
              <p>Control laser projectors.</p>
              <Link to="/laser-control" className="btn btn-primary">Go to Laser Control</Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Data Collection component
function DataCollection({ socket }) {
  return (
    <div className="module-section">
      <h2>Data Collection</h2>
      <p>Collect data from solar panels.</p>
      
      {/* Placeholder for data collection interface */}
      <div className="visualization">
        <p className="text-center pt-5">Solar Data Visualization</p>
      </div>
    </div>
  );
}

// Audio Conversion component
function AudioConversion({ socket }) {
  return (
    <div className="module-section">
      <h2>Audio Conversion</h2>
      <p>Convert solar data to audio.</p>
      
      {/* Placeholder for audio conversion interface */}
      <div className="visualization">
        <p className="text-center pt-5">Audio Waveform Visualization</p>
      </div>
    </div>
  );
}

// RAVE Processing component
function RaveProcessing({ socket }) {
  return (
    <div className="module-section">
      <h2>RAVE Processing</h2>
      <p>Process audio using RAVE.</p>
      
      {/* Placeholder for RAVE processing interface */}
      <div className="visualization">
        <p className="text-center pt-5">RAVE Processing Visualization</p>
      </div>
    </div>
  );
}

// Vector Generation component
function VectorGeneration({ socket }) {
  return (
    <div className="module-section">
      <h2>Vector Generation</h2>
      <p>Generate vector graphics from processed audio.</p>
      
      {/* Placeholder for vector generation interface */}
      <div className="visualization">
        <p className="text-center pt-5">Vector Graphics Visualization</p>
      </div>
    </div>
  );
}

// Laser Control component
function LaserControl({ socket }) {
  return (
    <div className="module-section">
      <h2>Laser Control</h2>
      <p>Control laser projectors.</p>
      
      {/* Placeholder for laser control interface */}
      <div className="visualization">
        <p className="text-center pt-5">Laser Projection Simulation</p>
      </div>
    </div>
  );
}

export default App;
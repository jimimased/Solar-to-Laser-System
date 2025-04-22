/**
 * Web server for the Solar-to-Laser System.
 */

const express = require('express');
const path = require('path');
const http = require('http');
const socketIo = require('socket.io');
const axios = require('axios');

// Create Express app
const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Environment variables
const PORT = process.env.PORT || 80;
const DATA_COLLECTION_URL = process.env.DATA_COLLECTION_URL || 'http://localhost:8000';
const AUDIO_CONVERSION_URL = process.env.AUDIO_CONVERSION_URL || 'http://localhost:8001';
const RAVE_PROCESSING_URL = process.env.RAVE_PROCESSING_URL || 'http://localhost:8002';
const VECTOR_GENERATION_URL = process.env.VECTOR_GENERATION_URL || 'http://localhost:8003';
const LASER_CONTROL_URL = process.env.LASER_CONTROL_URL || 'http://localhost:8004';

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Parse JSON request body
app.use(express.json());

// API proxy routes
app.use('/api/solar', createProxy(DATA_COLLECTION_URL));
app.use('/api/audio', createProxy(AUDIO_CONVERSION_URL));
app.use('/api/rave', createProxy(RAVE_PROCESSING_URL));
app.use('/api/vector', createProxy(VECTOR_GENERATION_URL));
app.use('/api/laser', createProxy(LASER_CONTROL_URL));

// Serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Socket.IO connection
io.on('connection', (socket) => {
  console.log('Client connected');
  
  // Handle disconnect
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
  
  // Handle solar data
  socket.on('solar_data', (data) => {
    // Forward to data collection API
    axios.post(`${DATA_COLLECTION_URL}/api/solar/data`, data)
      .then(response => {
        socket.emit('solar_data_response', response.data);
      })
      .catch(error => {
        socket.emit('error', { message: error.message });
      });
  });
  
  // Handle audio conversion
  socket.on('convert_audio', (data) => {
    // Forward to audio conversion API
    axios.post(`${AUDIO_CONVERSION_URL}/api/audio/convert`, data)
      .then(response => {
        socket.emit('audio_conversion_response', response.data);
      })
      .catch(error => {
        socket.emit('error', { message: error.message });
      });
  });
  
  // Handle RAVE processing
  socket.on('process_rave', (data) => {
    // Forward to RAVE processing API
    axios.post(`${RAVE_PROCESSING_URL}/api/rave/process`, data)
      .then(response => {
        socket.emit('rave_processing_response', response.data);
      })
      .catch(error => {
        socket.emit('error', { message: error.message });
      });
  });
  
  // Handle vector generation
  socket.on('generate_vector', (data) => {
    // Forward to vector generation API
    axios.post(`${VECTOR_GENERATION_URL}/api/vector/generate`, data)
      .then(response => {
        socket.emit('vector_generation_response', response.data);
      })
      .catch(error => {
        socket.emit('error', { message: error.message });
      });
  });
  
  // Handle laser control
  socket.on('control_laser', (data) => {
    // Forward to laser control API
    axios.post(`${LASER_CONTROL_URL}/api/laser/generate`, data)
      .then(response => {
        socket.emit('laser_control_response', response.data);
      })
      .catch(error => {
        socket.emit('error', { message: error.message });
      });
  });
});

// Create proxy middleware
function createProxy(target) {
  return async (req, res) => {
    try {
      const url = `${target}${req.url}`;
      const method = req.method.toLowerCase();
      
      const response = await axios({
        method,
        url,
        data: method !== 'get' ? req.body : undefined,
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      res.status(response.status).json(response.data);
    } catch (error) {
      console.error(`Proxy error: ${error.message}`);
      res.status(error.response?.status || 500).json({
        error: error.message,
      });
    }
  };
}

// Start server
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
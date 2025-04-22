"""
Tests for the laser control module.
"""

import unittest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.laser_control.controller import (
    LaserController,
    ILDAController,
    PangolinController,
    SimulationController
)
from src.laser_control.ilda import (
    ILDAFile,
    convert_svg_to_ilda,
    parse_svg_path
)


class TestLaserController(unittest.TestCase):
    """Tests for the LaserController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock device
        self.mock_device = MagicMock()
        
        # Initialize the laser controller
        self.controller = LaserController(device=self.mock_device)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_connect(self):
        """Test connecting to the laser device."""
        # Connect to the device
        self.controller.connect()
        
        # Assert that the device.connect method was called
        self.mock_device.connect.assert_called_once()
    
    def test_disconnect(self):
        """Test disconnecting from the laser device."""
        # Disconnect from the device
        self.controller.disconnect()
        
        # Assert that the device.disconnect method was called
        self.mock_device.disconnect.assert_called_once()
    
    def test_send_frame(self):
        """Test sending a frame to the laser device."""
        # Create a mock frame
        frame = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
            [0.5, 0.5, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0, 1.0]
        ])
        
        # Send the frame
        self.controller.send_frame(frame)
        
        # Assert that the device.send_frame method was called with the correct frame
        self.mock_device.send_frame.assert_called_once()
        np.testing.assert_array_equal(self.mock_device.send_frame.call_args[0][0], frame)
    
    def test_send_frames(self):
        """Test sending multiple frames to the laser device."""
        # Create mock frames
        frames = [
            np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
                [0.5, 0.5, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0]
            ]),
            np.array([
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0, 1.0]
            ])
        ]
        
        # Send the frames
        self.controller.send_frames(frames)
        
        # Assert that the device.send_frame method was called twice with the correct frames
        self.assertEqual(self.mock_device.send_frame.call_count, 2)
        np.testing.assert_array_equal(self.mock_device.send_frame.call_args_list[0][0][0], frames[0])
        np.testing.assert_array_equal(self.mock_device.send_frame.call_args_list[1][0][0], frames[1])
    
    def test_play_ilda_file(self):
        """Test playing an ILDA file."""
        # Create a mock ILDA file
        ilda_file = MagicMock()
        ilda_file.frames = [
            np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
                [0.5, 0.5, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0]
            ]),
            np.array([
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0, 1.0]
            ])
        ]
        
        # Play the ILDA file
        self.controller.play_ilda_file(ilda_file)
        
        # Assert that the device.send_frame method was called twice with the correct frames
        self.assertEqual(self.mock_device.send_frame.call_count, 2)
        np.testing.assert_array_equal(self.mock_device.send_frame.call_args_list[0][0][0], ilda_file.frames[0])
        np.testing.assert_array_equal(self.mock_device.send_frame.call_args_list[1][0][0], ilda_file.frames[1])


class TestILDAController(unittest.TestCase):
    """Tests for the ILDAController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock device
        self.mock_device = MagicMock()
        
        # Initialize the ILDA controller
        self.controller = ILDAController(device=self.mock_device)
    
    def test_send_frame(self):
        """Test sending a frame to the ILDA device."""
        # Create a mock frame
        frame = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
            [0.5, 0.5, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0]
        ])
        
        # Send the frame
        self.controller.send_frame(frame)
        
        # Assert that the device.write method was called with the correct data
        self.mock_device.write.assert_called_once()
        
        # The data sent to the device should be a binary representation of the frame
        sent_data = self.mock_device.write.call_args[0][0]
        self.assertIsInstance(sent_data, bytes)


class TestPangolinController(unittest.TestCase):
    """Tests for the PangolinController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock device
        self.mock_device = MagicMock()
        
        # Initialize the Pangolin controller
        self.controller = PangolinController(device=self.mock_device)
    
    def test_send_frame(self):
        """Test sending a frame to the Pangolin device."""
        # Create a mock frame
        frame = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
            [0.5, 0.5, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0]
        ])
        
        # Send the frame
        self.controller.send_frame(frame)
        
        # Assert that the device.send_frame method was called with the correct frame
        self.mock_device.send_frame.assert_called_once()
        
        # The data sent to the device should be a Pangolin-specific representation of the frame
        sent_data = self.mock_device.send_frame.call_args[0][0]
        self.assertIsInstance(sent_data, np.ndarray)


class TestSimulationController(unittest.TestCase):
    """Tests for the SimulationController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize the simulation controller
        self.controller = SimulationController()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_send_frame(self):
        """Test sending a frame to the simulation."""
        # Create a mock frame
        frame = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
            [0.5, 0.5, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0]
        ])
        
        # Send the frame
        self.controller.send_frame(frame)
        
        # Assert that the frame was stored in the controller
        self.assertEqual(len(self.controller.frames), 1)
        np.testing.assert_array_equal(self.controller.frames[0], frame)
    
    def test_save_simulation(self):
        """Test saving the simulation to a file."""
        # Create mock frames
        frames = [
            np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
                [0.5, 0.5, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0]
            ]),
            np.array([
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0, 1.0]
            ])
        ]
        
        # Send the frames
        for frame in frames:
            self.controller.send_frame(frame)
        
        # Save the simulation
        output_path = os.path.join(self.output_dir, "simulation.mp4")
        self.controller.save_simulation(output_path)
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))


class TestILDAFile(unittest.TestCase):
    """Tests for the ILDAFile class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize the ILDA file
        self.ilda_file = ILDAFile()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_add_frame(self):
        """Test adding a frame to the ILDA file."""
        # Create a mock frame
        frame = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
            [0.5, 0.5, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0]
        ])
        
        # Add the frame
        self.ilda_file.add_frame(frame)
        
        # Assert that the frame was added
        self.assertEqual(len(self.ilda_file.frames), 1)
        np.testing.assert_array_equal(self.ilda_file.frames[0], frame)
    
    def test_save_and_load(self):
        """Test saving and loading an ILDA file."""
        # Create mock frames
        frames = [
            np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],  # X, Y, R, G, B
                [0.5, 0.5, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0]
            ]),
            np.array([
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0, 1.0]
            ])
        ]
        
        # Add the frames
        for frame in frames:
            self.ilda_file.add_frame(frame)
        
        # Save the ILDA file
        output_path = os.path.join(self.output_dir, "test.ild")
        self.ilda_file.save(output_path)
        
        # Assert that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load the ILDA file
        loaded_file = ILDAFile()
        loaded_file.load(output_path)
        
        # Assert that the loaded file has the correct number of frames
        self.assertEqual(len(loaded_file.frames), len(frames))


class TestSVGToILDA(unittest.TestCase):
    """Tests for the SVG to ILDA conversion functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create a simple SVG file
        self.svg_path = os.path.join(self.output_dir, "test.svg")
        with open(self.svg_path, "w") as f:
            f.write("""
            <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
                <path d="M 10,10 L 90,10 L 90,90 L 10,90 Z" fill="none" stroke="black" />
            </svg>
            """)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_parse_svg_path(self):
        """Test parsing an SVG path."""
        # Parse the SVG path
        path_str = "M 10,10 L 90,10 L 90,90 L 10,90 Z"
        points = parse_svg_path(path_str)
        
        # Assert that the points have the correct shape
        self.assertGreater(len(points), 0)
        self.assertEqual(points.shape[1], 2)  # X and Y coordinates
    
    def test_convert_svg_to_ilda(self):
        """Test converting an SVG file to ILDA format."""
        # Convert the SVG file to ILDA
        ilda_file = convert_svg_to_ilda(self.svg_path)
        
        # Assert that the ILDA file has at least one frame
        self.assertGreater(len(ilda_file.frames), 0)
        
        # Assert that the frame has the correct shape
        frame = ilda_file.frames[0]
        self.assertGreater(len(frame), 0)
        self.assertEqual(frame.shape[1], 5)  # X, Y, R, G, B


if __name__ == "__main__":
    unittest.main()
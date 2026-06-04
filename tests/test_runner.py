import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call

import numpy as np

# Add parent directory to path so we can import danger_zone_monitor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock out postgres_db to avoid connecting to real database during unit tests
from unittest.mock import MagicMock
mock_db = MagicMock()
mock_db.create_default_intrusion_event.return_value = None
mock_db.create_default_loitering_alert.return_value = None
mock_db.create_default_line_crossing.return_value = None
sys.modules['postgres_db'] = mock_db

from danger_zone_monitor.models import Zone, TrackedPerson, IntrusionEvent
from danger_zone_monitor.zone_manager import ZoneManager
from danger_zone_monitor.csv_logger import CSVLogger
from danger_zone_monitor.intrusion_manager import IntrusionManager, PersonZoneState
from danger_zone_monitor.video_recorder import VideoRecorder
from danger_zone_monitor.person_tracker import BoTSORTLiteTracker, Detection


class TestModels(unittest.TestCase):
    """Tests properties and post-initialization of core data models."""

    def test_zone_creation(self):
        points = [(10, 10), (100, 10), (100, 100), (10, 100)]
        zone = Zone(zone_id=1, zone_name="TestZone", points=points)
        self.assertEqual(zone.zone_id, 1)
        self.assertEqual(zone.zone_name, "TestZone")
        self.assertEqual(zone.polygon_np.shape, (4, 2))
        self.assertEqual(zone.polygon_np.dtype, np.int32)

    def test_tracked_person_center(self):
        # Bbox: (x1, y1, x2, y2)
        person = TrackedPerson(track_id=42, bbox=(10.0, 20.0, 50.0, 100.0), confidence=0.85)
        self.assertEqual(person.track_id, 42)
        # cx = (10 + 50) / 2 = 30
        # cy = (20 + 100) / 2 = 60
        self.assertEqual(person.center, (30, 60))


class TestZoneManager(unittest.TestCase):
    """Tests loading of configuration coordinates and point-in-polygon checks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "zones.json")
        
        # Write mock zones configuration
        mock_config = {
            "zones": [
                {
                    "zone_id": 1,
                    "zone_name": "BoxZone",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]]
                }
            ]
        }
        with open(self.config_path, "w") as f:
            json.dump(mock_config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_zones(self):
        zm = ZoneManager(self.config_path)
        self.assertEqual(len(zm.zones), 1)
        self.assertEqual(zm.zones[0].zone_name, "BoxZone")
        self.assertEqual(zm.zones[0].zone_id, 1)

    def test_check_point_in_zones(self):
        zm = ZoneManager(self.config_path)
        # Point inside
        inside = zm.check_point_in_zones((5, 5))
        self.assertEqual(len(inside), 1)
        self.assertEqual(inside[0].zone_id, 1)
        
        # Point outside
        outside = zm.check_point_in_zones((15, 5))
        self.assertEqual(len(outside), 0)


class TestCSVLogger(unittest.TestCase):
    """Tests CSV file initialization and logging format."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_log.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_csv_headers_and_write(self):
        logger = CSVLogger(self.log_path)
        # Verify file and headers
        self.assertTrue(os.path.exists(self.log_path))
        with open(self.log_path, "r") as f:
            header = f.readline().strip()
            self.assertEqual(header, "EventID,PersonID,ZoneID,ZoneName,EntryTime,ExitTime,DurationSeconds,VideoPath,SnapshotPath")
            
        # Log an event
        t_entry = datetime(2026, 6, 1, 12, 0, 0)
        t_exit = datetime(2026, 6, 1, 12, 0, 5)
        logger.log_event(
            event_id=1,
            person_id=42,
            zone_id=10,
            zone_name="TestZone",
            entry_time=t_entry,
            exit_time=t_exit,
            duration=5.0,
            video_path="recordings/evt1.mp4",
            snapshot_path="snapshots/evt1.jpg"
        )
        
        with open(self.log_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + Data
            data_row = lines[1].strip().split(",")
            self.assertEqual(data_row[0], "1")
            self.assertEqual(data_row[1], "42")
            self.assertEqual(data_row[2], "10")
            self.assertEqual(data_row[3], "TestZone")
            self.assertEqual(data_row[6], "5.00")
            self.assertTrue(data_row[7].endswith("evt1.mp4"))
            self.assertTrue(data_row[8].endswith("evt1.jpg"))


class TestVideoRecorder(unittest.TestCase):
    """Tests event file naming behavior."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recordings_dir = os.path.join(self.temp_dir, "recordings")
        self.snapshots_dir = os.path.join(self.temp_dir, "snapshots")
        self.recorder = VideoRecorder(
            recordings_dir=self.recordings_dir,
            snapshots_dir=self.snapshots_dir
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_next_event_id_scans_existing_outputs(self):
        # Files now live inside subdirectories
        open(os.path.join(self.recordings_dir, "danger_zone", "event_0005.mp4"), "w").close()
        open(os.path.join(self.recordings_dir, "danger_zone", "event_0008_part_002.mp4"), "w").close()
        open(os.path.join(self.snapshots_dir, "loitering", "event_0007.jpg"), "w").close()

        self.assertEqual(self.recorder.get_next_event_id(), 9)

    def test_save_snapshot_with_suffix(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        path = self.recorder.save_snapshot(frame, 42, suffix="_loiter_10s", subdir="loitering")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("event_0042_loiter_10s.jpg"))
        self.assertIn("loitering", path)


class TestBoTSORTLiteTracker(unittest.TestCase):
    """Tests lightweight BoT-SORT-style ID continuity."""

    def test_preserves_track_id_with_motion_prediction(self):
        tracker = BoTSORTLiteTracker(iou_threshold=0.2, max_missed=3)

        first = tracker.update([
            Detection(bbox=(10, 10, 50, 90), confidence=0.9)
        ])
        self.assertEqual(len(first), 1)
        track_id = first[0].track_id

        second = tracker.update([
            Detection(bbox=(14, 10, 54, 90), confidence=0.88)
        ])
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].track_id, track_id)

        missing = tracker.update([])
        self.assertEqual(missing, [])

        third = tracker.update([
            Detection(bbox=(20, 10, 60, 90), confidence=0.86)
        ])
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0].track_id, track_id)


class TestIntrusionManager(unittest.TestCase):
    """Tests temporal state transitions, lost tracks buffering, and exit triggers."""

    def setUp(self):
        # Mocks
        self.mock_recorder = MagicMock(spec=VideoRecorder)
        self.mock_recorder.get_next_event_id.return_value = 1
        self.mock_recorder.save_snapshot.return_value = "snapshots/danger_zone/mock.jpg"
        self.mock_recorder.start_recording.return_value = (MagicMock(), "recordings/danger_zone/mock.mp4")
        self.mock_recorder.recordings_dir = "recordings"
        self.mock_recorder.snapshots_dir = "snapshots"
        
        self.mock_csv = MagicMock(spec=CSVLogger)
        self.mock_loitering_csv = MagicMock(spec=CSVLogger)
        
        self.fps = 10.0
        self.im = IntrusionManager(
            video_recorder=self.mock_recorder,
            csv_logger=self.mock_csv,
            loitering_csv_logger=self.mock_loitering_csv,
            fps=self.fps
        )
        
        # Setup mock zones: Zone 1
        self.mock_zone = Zone(zone_id=1, zone_name="DangerA", points=[(0, 0), (10, 0), (10, 10), (0, 10)])
        self.mock_zm = MagicMock(spec=ZoneManager)
        self.mock_zm.zones = [self.mock_zone]
        
        # Frame mock (100x100 RGB)
        self.mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_unconfirmed_intrusion_exit(self):
        # 1. Person enters zone
        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        person = TrackedPerson(track_id=1, bbox=(2, 2, 8, 8), confidence=0.9) # Bottom center (5, 8)
        
        # Frame 1
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        key = (1, 1)
        self.assertIn(key, self.im.active_states)
        self.assertFalse(self.im.active_states[key].is_confirmed)
        
        # 2. Person exits immediately in frame 2
        self.mock_zm.check_point_in_zones.return_value = []
        # Run update 31 times to exceed MAX_MISSING_FRAMES (30)
        for _ in range(35):
            self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
            
        # Assert state is cleared because duration was under 3 seconds and track exited
        self.assertNotIn(key, self.im.active_states)
        self.mock_recorder.start_recording.assert_not_called()
        self.mock_csv.log_event.assert_not_called()

    def test_confirmed_intrusion_process(self):
        # 1. Person enters zone
        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        person = TrackedPerson(track_id=2, bbox=(2, 2, 8, 8), confidence=0.9)
        
        # Frame 1
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        key = (2, 1)
        
        # 2. Simulating a delay >= 3.0s by manually adjusting entry time backwards
        # (Since we are testing temporal logic, we modify entry_time to mock 4 seconds inside)
        self.im.active_states[key].entry_time = datetime.now() - timedelta(seconds=4)
        
        # Update with frame again to trigger confirmation
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        
        # Verify confirmed status and video initialization
        self.assertTrue(self.im.active_states[key].is_confirmed)
        self.assertEqual(self.im.active_states[key].event_id, 1)
        self.mock_recorder.save_snapshot.assert_called_once()
        self.mock_recorder.start_recording.assert_called_once()
        
        # 3. Simulate track exit (above missing frame threshold)
        self.mock_zm.check_point_in_zones.return_value = []
        for _ in range(35):
            self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
            
        # State should be removed
        self.assertNotIn(key, self.im.active_states)
        
        # Video stopped and details written to CSV
        self.mock_recorder.stop_recording.assert_called_once()
        self.mock_csv.log_event.assert_called_once()

    def test_missing_frames_tolerance(self):
        # Person enters zone
        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        person = TrackedPerson(track_id=3, bbox=(2, 2, 8, 8), confidence=0.9)
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        key = (3, 1)
        
        # Disappear for 10 frames
        self.mock_zm.check_point_in_zones.return_value = []
        for _ in range(10):
            self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
            
        # Assert state is still active due to tolerance
        self.assertIn(key, self.im.active_states)
        self.assertEqual(self.im.active_states[key].missing_frames, 10)
        
        # Person reappears in zone
        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        
        # Missing frames should reset to 0
        self.assertEqual(self.im.active_states[key].missing_frames, 0)

    def test_confirmed_intrusion_splits_recording_segments(self):
        self.im.max_segment_frames = 2
        writer_1 = MagicMock()
        writer_2 = MagicMock()
        self.mock_recorder.start_recording.side_effect = [
            (writer_1, "recordings/event_0001.mp4"),
            (writer_2, "recordings/event_0001_part_002.mp4")
        ]

        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        person = TrackedPerson(track_id=4, bbox=(2, 2, 8, 8), confidence=0.9)

        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        key = (4, 1)
        self.im.active_states[key].entry_time = datetime.now() - timedelta(seconds=4)
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)

        state = self.im.active_states[key]
        self.assertEqual(state.segment_index, 2)
        self.assertEqual(state.segment_frame_count, 1)
        self.assertEqual(
            state.video_paths,
            ["recordings/event_0001.mp4", "recordings/event_0001_part_002.mp4"]
        )
        self.mock_recorder.start_recording.assert_has_calls([
            call(event_id=1, fps=self.fps, frame_size=(100, 100), subdir="danger_zone"),
            call(event_id=1, fps=self.fps, frame_size=(100, 100), subdir="danger_zone", segment_index=2)
        ])
        self.mock_recorder.stop_recording.assert_called_once_with(writer_1)

    def test_starts_from_next_available_event_id(self):
        self.mock_recorder.get_next_event_id.return_value = 6
        manager = IntrusionManager(
            video_recorder=self.mock_recorder,
            csv_logger=self.mock_csv,
            loitering_csv_logger=self.mock_loitering_csv,
            fps=self.fps
        )

        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        person = TrackedPerson(track_id=4, bbox=(2, 2, 8, 8), confidence=0.9)

        manager.update_with_frame([person], self.mock_zm, self.mock_frame)
        key = (4, 1)
        manager.active_states[key].entry_time = datetime.now() - timedelta(seconds=4)
        manager.update_with_frame([person], self.mock_zm, self.mock_frame)

        self.assertEqual(manager.active_states[key].event_id, 6)
        self.assertEqual(manager.next_event_id, 7)
        self.mock_recorder.start_recording.assert_called_with(
            event_id=6,
            fps=self.fps,
            frame_size=(100, 100),
            subdir="danger_zone"
        )

    def test_loitering_detection_flow(self):
        self.im.loitering_threshold = 2.0
        self.im.loitering_alert_cooldown = 1.0

        # 1. Person enters zone
        self.mock_zm.check_point_in_zones.return_value = [self.mock_zone]
        person = TrackedPerson(track_id=5, bbox=(2, 2, 8, 8), confidence=0.9)

        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)
        key = (5, 1)

        # Confirm intrusion first (by modifying entry_time to 4s ago)
        self.im.active_states[key].entry_time = datetime.now() - timedelta(seconds=4)
        self.im.update_with_frame([person], self.mock_zm, self.mock_frame)

        # Verify intrusion is confirmed and is_loitering is True
        self.assertTrue(self.im.active_states[key].is_confirmed)
        self.assertTrue(self.im.active_states[key].is_loitering)

        # Check that save_snapshot was called with subdir params
        self.mock_recorder.save_snapshot.assert_has_calls([
            call(self.mock_frame, 1, subdir="danger_zone"),
            call(self.mock_frame, 1, suffix="_loiter_4s", subdir="loitering")
        ])


class TestPostgresDb(unittest.TestCase):
    """Tests for database operations, focusing on the newly added zone deletion function."""

    def test_delete_zone(self):
        import sys
        import importlib

        # Save the mocked version
        mocked_db = sys.modules.get('postgres_db')
        if 'postgres_db' in sys.modules:
            del sys.modules['postgres_db']

        try:
            # Import the real postgres_db module
            import postgres_db
            
            # Setup mocks for psycopg2 connection and cursor
            postgres_db.conn = MagicMock()
            postgres_db.cursor = MagicMock()
            
            # Mock ensure_connection to return True
            original_ensure = postgres_db.ensure_connection
            postgres_db.ensure_connection = MagicMock(return_value=True)
            
            try:
                # Call delete_zone
                postgres_db.delete_zone(42)
                
                # Check cursor.execute was called with correct SQL and args
                postgres_db.cursor.execute.assert_called_once_with("DELETE FROM zones WHERE id = %s;", (42,))
                # Check connection commit was called
                postgres_db.conn.commit.assert_called_once()
            finally:
                postgres_db.ensure_connection = original_ensure
        finally:
            # Restore the mock db for other tests
            if mocked_db:
                sys.modules['postgres_db'] = mocked_db


if __name__ == "__main__":
    unittest.main()

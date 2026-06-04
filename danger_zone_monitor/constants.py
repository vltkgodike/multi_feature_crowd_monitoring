"""Constants and configuration defaults for the danger zone monitor system."""

# The maximum number of consecutive frames a person's track can be missing
# before they are registered as exited.
MAX_MISSING_FRAMES: int = 30

# The minimum time in seconds a person must continuously remain in a zone
# to trigger a confirmed intrusion event.
CONFIRMATION_THRESHOLD_SEC: float = 3.0

# The duration of pre-recording buffer in seconds.
PRE_RECORD_SECONDS: float = 3.0

# Maximum length of each saved recording clip in seconds.
RECORDING_SEGMENT_SECONDS: float = 60.0

# Default FPS used if the frame rate cannot be determined dynamically.
DEFAULT_FPS: float = 30.0

# The minimum time in seconds a person must continuously remain in a zone
# to trigger a confirmed loitering event.
LOITERING_THRESHOLD_SEC: float = 10.0

# The cooldown time in seconds between successive loitering alerts for the same track.
LOITERING_ALERT_COOLDOWN_SEC: float = 10.0


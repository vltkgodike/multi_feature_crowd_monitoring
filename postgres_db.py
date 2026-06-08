
import sys

# Force stdout/stderr to encode in UTF-8 to prevent UnicodeEncodeError on Windows command prompt when printing emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from datetime import datetime, timedelta, timezone
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================================
# DATABASE CONFIG
# =========================================

DB_HOST = "aws-1-ap-northeast-2.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.vxpyljtmndedusttnvlz"
DB_PASSWORD = "Bhavanibab@30"
DB_PORT = 5432
APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "Asia/Kolkata")


# =========================================
# GLOBAL CONNECTION VARIABLES
# =========================================

conn = None
cursor = None


def _get_app_timezone():
    if ZoneInfo is not None:
        try:
            return ZoneInfo(APP_TIMEZONE)
        except Exception:
            pass

    if APP_TIMEZONE in ("Asia/Kolkata", "Asia/Calcutta", "IST"):
        return timezone(timedelta(hours=5, minutes=30))

    return datetime.now().astimezone().tzinfo


def current_db_timestamp():
    """Return local app time for TIMESTAMP WITHOUT TIME ZONE columns."""
    return datetime.now(_get_app_timezone()).replace(tzinfo=None)


def to_db_timestamp(value=None):
    if value is None:
        return current_db_timestamp()

    if value.tzinfo is None:
        return value

    return value.astimezone(_get_app_timezone()).replace(tzinfo=None)


# =========================================
# CONNECT DATABASE
# =========================================

def connect_db():

    global conn, cursor

    try:

        if conn is not None and conn.closed == 0:

            print(" Database already connected")
            return True

        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            sslmode="require"
        )

        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        try:
            cursor.execute("SET TIME ZONE %s;", (APP_TIMEZONE,))
            conn.commit()
        except Exception as tz_error:
            conn.rollback()
            print(f" Failed to set database timezone to {APP_TIMEZONE}: {tz_error}")

        print(" Connected to Supabase PostgreSQL")

        return True

    except Exception as e:

        print(" Database connection failed")
        print(e)

        conn = None
        cursor = None

        return False


# =========================================
# CHECK CONNECTION
# =========================================

def is_connected():

    global conn

    try:

        return conn is not None and conn.closed == 0

    except Exception:

        return False


# =========================================
# ENSURE CONNECTION
# =========================================

def ensure_connection():

    if not is_connected():

        print(" Reconnecting database...")

        return connect_db()

    return True


# =========================================
# CREATE ZONES TABLE
# =========================================

def create_zones_table():

    global conn, cursor

    try:

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS zones (

            id SERIAL PRIMARY KEY,

            name VARCHAR(255) DEFAULT '0',

            points JSONB DEFAULT '[]',

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        """)

        conn.commit()

        print(" zones table created")

    except Exception as e:

        print(" zones table creation failed")
        print(e)

        conn.rollback()


# =========================================
# CREATE INTRUSION EVENTS TABLE
# =========================================

def create_intrusion_events_table():

    global conn, cursor

    try:

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS intrusion_events (

            event_id SERIAL PRIMARY KEY,

            person_id INTEGER DEFAULT 0,

            zone_id INTEGER DEFAULT 0,

            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            exit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            duration_seconds FLOAT DEFAULT 0,

            video_path TEXT DEFAULT '0',

            snapshot_path TEXT DEFAULT '0',

            is_loitering BOOLEAN DEFAULT FALSE,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        """)

        conn.commit()

        print(" intrusion_events table created")

    except Exception as e:

        print(" intrusion_events table creation failed")
        print(e)

        conn.rollback()


# =========================================
# CREATE LOITERING ALERTS TABLE
# =========================================

def create_loitering_alerts_table():

    global conn, cursor

    try:

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS loitering_alerts (

            alert_id SERIAL PRIMARY KEY,

            event_id INTEGER DEFAULT 0,

            dwell_time_seconds FLOAT DEFAULT 0,

            alert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            snapshot_path TEXT DEFAULT '0',

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        """)

        conn.commit()

        print(" loitering_alerts table created")

    except Exception as e:

        print(" loitering_alerts table creation failed")
        print(e)

        conn.rollback()


# =========================================
# CREATE LINE CROSSINGS TABLE
# =========================================

def create_line_crossings_table():

    global conn, cursor

    try:

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS line_crossings (

            crossing_id SERIAL PRIMARY KEY,

            person_id INTEGER DEFAULT 0,

            direction VARCHAR(10) DEFAULT '0',

            crossing_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        """)

        conn.commit()

        print(" line_crossings table created")

    except Exception as e:

        print(" line_crossings table creation failed")
        print(e)

        conn.rollback()


# =========================================
# INSERT DEFAULT ZONE
# =========================================

def create_default_zone():

    global conn, cursor

    try:

        ensure_connection()

        cursor.execute(
            """

            INSERT INTO zones (

                name,
                points,
                created_at

            )

            VALUES (%s, %s, %s)

            RETURNING id;

            """,
            (
                "0",
                json.dumps([]),
                current_db_timestamp()
            )
        )

        result = cursor.fetchone()

        conn.commit()

        print(f" Default zone created ID: {result['id']}")

        return result["id"]

    except Exception as e:

        print(" Default zone creation failed")
        print(e)

        conn.rollback()

        return None


# =========================================
# UPDATE ZONE DATA FROM SERVER
# =========================================

def update_zone(zone_id, name, points):

    global conn, cursor

    try:

        ensure_connection()

        cursor.execute(
            """

            UPDATE zones

            SET
                name = %s,
                points = %s

            WHERE id = %s;

            """,
            (
                name,
                json.dumps(points),
                zone_id
            )
        )

        conn.commit()

        print(f" Zone updated ID: {zone_id}")

    except Exception as e:

        print(" Zone update failed")
        print(e)

        conn.rollback()


# =========================================
# INSERT DEFAULT INTRUSION EVENT
# =========================================

def create_default_intrusion_event(zone_id=0, entry_time=None):

    global conn, cursor

    try:

        ensure_connection()
        event_time = to_db_timestamp(entry_time)
        created_at = current_db_timestamp()

        cursor.execute(
            """

            INSERT INTO intrusion_events (
                person_id,
                zone_id,
                entry_time,
                exit_time,
                duration_seconds,
                video_path,
                snapshot_path,
                is_loitering,
                created_at
            )
            VALUES (0, %s, %s, %s, 0.0, '0', '0', FALSE, %s)

            RETURNING event_id;

            """,
            (zone_id, event_time, event_time, created_at)
        )

        result = cursor.fetchone()

        conn.commit()

        print(f" Default intrusion event created ID: {result['event_id']}")

        return result["event_id"]

    except Exception as e:

        print(" Default intrusion event creation failed")
        print(e)

        conn.rollback()

        return None


# =========================================
# UPDATE INTRUSION EVENT FROM SERVER
# =========================================

def update_intrusion_event(
    event_id,
    person_id,
    zone_id,
    duration_seconds,
    video_path,
    snapshot_path,
    is_loitering,
    exit_time=None
):

    global conn, cursor

    try:

        ensure_connection()

        cursor.execute(
            """

            UPDATE intrusion_events

            SET
                person_id = %s,
                zone_id = %s,
                exit_time = %s,
                duration_seconds = %s,
                video_path = %s,
                snapshot_path = %s,
                is_loitering = %s

            WHERE event_id = %s;

            """,
            (
                person_id,
                zone_id,
                to_db_timestamp(exit_time),
                duration_seconds,
                video_path,
                snapshot_path,
                is_loitering,
                event_id
            )
        )

        conn.commit()

        print(f" Intrusion event updated ID: {event_id}")

    except Exception as e:

        print(" Intrusion event update failed")
        print(e)

        conn.rollback()


# =========================================
# INSERT DEFAULT LOITERING ALERT
# =========================================

def create_default_loitering_alert(event_id, alert_time=None):

    global conn, cursor

    try:

        ensure_connection()
        alert_time = to_db_timestamp(alert_time)

        cursor.execute(
            """

            INSERT INTO loitering_alerts (
                event_id,
                dwell_time_seconds,
                alert_time,
                snapshot_path,
                created_at
            )
            VALUES (%s, 0.0, %s, '0', %s)

            RETURNING alert_id;

            """,
            (event_id, alert_time, alert_time)
        )

        result = cursor.fetchone()

        conn.commit()

        print(f" Default loitering alert created ID: {result['alert_id']}")

        return result["alert_id"]

    except Exception as e:

        print(" Default loitering alert creation failed")
        print(e)

        conn.rollback()

        return None


# =========================================
# UPDATE LOITERING ALERT FROM SERVER
# =========================================

def update_loitering_alert(
    alert_id,
    event_id,
    dwell_time_seconds,
    snapshot_path,
    alert_time=None
):

    global conn, cursor

    try:

        ensure_connection()

        cursor.execute(
            """

            UPDATE loitering_alerts

            SET
                event_id = %s,
                dwell_time_seconds = %s,
                alert_time = %s,
                snapshot_path = %s

            WHERE alert_id = %s;

            """,
            (
                event_id,
                dwell_time_seconds,
                to_db_timestamp(alert_time),
                snapshot_path,
                alert_id
            )
        )

        conn.commit()

        print(f" Default loitering alert updated ID: {alert_id}")

    except Exception as e:

        print(" Default loitering alert update failed")
        print(e)

        conn.rollback()


# =========================================
# INSERT DEFAULT LINE CROSSING
# =========================================

def create_default_line_crossing(direction="IN", crossing_time=None):

    global conn, cursor

    try:

        ensure_connection()
        crossing_time = to_db_timestamp(crossing_time)

        cursor.execute(
            """

            INSERT INTO line_crossings (
                person_id,
                direction,
                crossing_time,
                created_at
            )
            VALUES (0, %s, %s, %s)

            RETURNING crossing_id;

            """,
            (direction, crossing_time, crossing_time)
        )

        result = cursor.fetchone()

        conn.commit()

        print(f" Default line crossing created ID: {result['crossing_id']}")

        return result["crossing_id"]

    except Exception as e:

        print(" Default line crossing creation failed")
        print(e)

        conn.rollback()

        return None


# =========================================
# UPDATE LINE CROSSING FROM SERVER
# =========================================

def update_line_crossing(
    crossing_id,
    person_id,
    direction,
    crossing_time=None
):

    global conn, cursor

    try:

        ensure_connection()

        cursor.execute(
            """

            UPDATE line_crossings

            SET
                person_id = %s,
                direction = %s,
                crossing_time = %s

            WHERE crossing_id = %s;

            """,
            (
                person_id,
                direction,
                to_db_timestamp(crossing_time),
                crossing_id
            )
        )

        conn.commit()

        print(f" Default line crossing updated ID: {crossing_id}")

    except Exception as e:

        print(" Default line crossing update failed")
        print(e)

        conn.rollback()


# =========================================
# INITIALIZE DATABASE
# =========================================

def ensure_default_zones():
    """Creates a default camera-wide zone with ID 0 in the database if it doesn't exist."""
    global conn, cursor
    try:
        ensure_connection()
        cursor.execute(
            """
            INSERT INTO zones (id, name, points, created_at)
            VALUES (0, 'camera_wide', '[]'::jsonb, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO NOTHING;
            """
        )
        conn.commit()
        print(" Camera-wide default zone (ID: 0) verified/created.")
    except Exception as e:
        print(" Failed to create camera-wide default zone (ID: 0):", e)
        if conn:
            conn.rollback()


def init_db():
    """Initializes the database connection and creates all required tables if they don't exist."""
    if connect_db():
        create_zones_table()
        create_intrusion_events_table()
        create_loitering_alerts_table()
        create_line_crossings_table()
        ensure_default_zones()
        return True
    return False


# =========================================
# UPSERT ZONE DATA
# =========================================

def upsert_zone(zone_id, name, points):
    """Inserts a zone with specific ID if it doesn't exist, or updates it if it does."""
    global conn, cursor

    try:
        ensure_connection()

        # Check if zone with zone_id exists
        cursor.execute("SELECT id FROM zones WHERE id = %s;", (zone_id,))
        row = cursor.fetchone()

        if row:
            # Update existing zone
            update_zone(zone_id, name, points)
        else:
            # Insert new zone with specific ID
            cursor.execute(
                """
                INSERT INTO zones (
                    id,
                    name,
                    points,
                    created_at
                )
                VALUES (%s, %s, %s, %s);
                """,
                (
                    zone_id,
                    name,
                    json.dumps(points),
                    current_db_timestamp()
                )
            )
            conn.commit()
            print(f" Zone inserted ID: {zone_id}")

    except Exception as e:
        print(f" Zone upsert failed for ID: {zone_id}")
        print(e)
        if conn:
            conn.rollback()


# =========================================
# DELETE ZONE DATA
# =========================================

def delete_zone(zone_id):
    """Deletes a zone with specific ID if it exists."""
    global conn, cursor

    try:
        ensure_connection()

        cursor.execute("DELETE FROM zones WHERE id = %s;", (zone_id,))
        conn.commit()
        print(f" Zone deleted from DB ID: {zone_id}")

    except Exception as e:
        print(f" Zone deletion failed for ID: {zone_id}")
        print(e)
        if conn:
            conn.rollback()


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    connect_db()

    create_zones_table()

    create_intrusion_events_table()

    create_loitering_alerts_table()

    create_line_crossings_table()


    # =========================================
    # CREATE DEFAULT ROWS
    # =========================================

    zone_id = create_default_zone()

    event_id = create_default_intrusion_event(zone_id)

    alert_id = create_default_loitering_alert(event_id)

    crossing_id = create_default_line_crossing("IN")


    # =========================================
    # SERVER UPDATE EXAMPLES
    # =========================================

    update_zone(
        zone_id,
        "Danger Zone",
        [[100, 100], [200, 200]]
    )

    update_intrusion_event(
        event_id,
        101,
        zone_id,
        45.5,
        "/videos/video.mp4",
        "/snapshots/image.jpg",
        True
    )

    update_loitering_alert(
        alert_id,
        event_id,
        60.5,
        "/snapshots/alert.jpg"
    )

    update_line_crossing(
        crossing_id,
        101,
        "IN"
    )


    # =========================================
    # CLOSE CONNECTION
    # =========================================

    if cursor:
        cursor.close()

    if conn:
        conn.close()

    print(" Database setup completed")


#!/bin/bash
 
# ------------------------------------------------

# SERVER CONFIGURATION

# ------------------------------------------------
 
SERVER_USER="storage"

SERVER_IP="storage.valkontek.com"

SERVER_PORT="2222"
 
LOG_FILE="/home/sai/jetson-storage/projects/multi_feature_crowd_monitoring/video_sync.log"
 
STORAGE_THRESHOLD=50
 
# ------------------------------------------------

# SOURCE FOLDERS (JETSON)

# ------------------------------------------------
 
VIDEO_SOURCE="/home/sai/jetson-storage/projects/multi_feature_crowd_monitoring/recordings/"

SNAPSHOT_SOURCE="/home/sai/jetson-storage/projects/multi_feature_crowd_monitoring/snapshots/"
 
# ------------------------------------------------

# DESTINATION FOLDERS (SERVER)

# ------------------------------------------------
 
VIDEO_DEST="/home/storage/crowd_monitoring_storage/recordings/"

SNAPSHOT_DEST="/home/storage/crowd_monitoring_storage/snapshots/"
 
# ------------------------------------------------

# START LOG

# ------------------------------------------------
 
echo "====================================" >> "$LOG_FILE"

echo "Run Time: $(date)" >> "$LOG_FILE"
 
# ------------------------------------------------

# SYNC RECORDINGS

# ------------------------------------------------
 
echo "Syncing recordings..." >> "$LOG_FILE"
 
rsync -avz \

-e "ssh -p $SERVER_PORT" \

--remove-source-files \

--include="*/" \

--include="*.mp4" \

--include="*.avi" \

--include="*.mkv" \

--exclude="*" \

"$VIDEO_SOURCE" \

"${SERVER_USER}@${SERVER_IP}:${VIDEO_DEST}" \
>> "$LOG_FILE" 2>&1
 
# Remove empty directories

find "$VIDEO_SOURCE" -type d -empty -delete
 
# ------------------------------------------------

# SYNC SNAPSHOTS

# ------------------------------------------------
 
echo "Syncing snapshots..." >> "$LOG_FILE"
 
rsync -avz \

-e "ssh -p $SERVER_PORT" \

--remove-source-files \

--include="*/" \

--include="*.jpg" \

--include="*.jpeg" \

--include="*.png" \

--exclude="*" \

"$SNAPSHOT_SOURCE" \

"${SERVER_USER}@${SERVER_IP}:${SNAPSHOT_DEST}" \
>> "$LOG_FILE" 2>&1
 
# Remove empty directories

find "$SNAPSHOT_SOURCE" -type d -empty -delete
 
# ------------------------------------------------

# CHECK STORAGE USAGE

# ------------------------------------------------
 
USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
 
echo "Current Storage Usage: ${USAGE}%" >> "$LOG_FILE"
 
# ------------------------------------------------

# CLEANUP IF STORAGE > 50%

# ------------------------------------------------
 
if [ "$USAGE" -ge "$STORAGE_THRESHOLD" ]; then
 
    echo "Storage exceeded ${STORAGE_THRESHOLD}%." >> "$LOG_FILE"

    echo "Deleting files older than 7 days..." >> "$LOG_FILE"
 
    # Delete old recordings

    find "$VIDEO_SOURCE" \

    -type f \

    \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" \) \

    -mtime +7 \

    -delete
 
    # Delete old snapshots

    find "$SNAPSHOT_SOURCE" \

    -type f \

    \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) \

    -mtime +7 \

    -delete
 
    # Remove empty directories

    find "$VIDEO_SOURCE" -type d -empty -delete

    find "$SNAPSHOT_SOURCE" -type d -empty -delete
 
    echo "Cleanup completed." >> "$LOG_FILE"
 
fi
 
echo "Completed at: $(date)" >> "$LOG_FILE"

echo "" >> "$LOG_FILE"
 

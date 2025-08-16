#!/bin/bash

# Water Access Monte Carlo - Progress Monitoring Script
# This script provides real-time monitoring of the simulation progress

set -e

# Configuration
PROJECT_ID="water-access-compute"
ZONE="us-central1-a"
INSTANCE_NAME="water-access-compute-spot"
BUCKET_NAME="water-access-data"
TOTAL_ITERATIONS=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}        Water Access Monte Carlo - Progress Monitor${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

print_status() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_instance_status() {
    STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
    
    case $STATUS in
        RUNNING)
            echo -e "${GREEN}● Instance Status: RUNNING${NC}"
            return 0
            ;;
        TERMINATED|STOPPED)
            echo -e "${YELLOW}● Instance Status: $STATUS (Spot preemption or stopped)${NC}"
            return 1
            ;;
        STOPPING|PROVISIONING|STAGING)
            echo -e "${BLUE}● Instance Status: $STATUS${NC}"
            return 0
            ;;
        NOT_FOUND)
            echo -e "${RED}● Instance Status: NOT FOUND${NC}"
            return 1
            ;;
        *)
            echo -e "${YELLOW}● Instance Status: $STATUS${NC}"
            return 0
            ;;
    esac
}

get_instance_details() {
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
        MACHINE_TYPE=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(machineType.scope(machineTypes))")
        CREATED=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(creationTimestamp)")
        
        echo -e "${BLUE}● Machine Type:${NC} $MACHINE_TYPE"
        echo -e "${BLUE}● Created:${NC} $CREATED"
        
        # Calculate uptime
        if [ "$STATUS" == "RUNNING" ]; then
            START_EPOCH=$(date -d "$CREATED" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$CREATED" +%s)
            NOW_EPOCH=$(date +%s)
            UPTIME_SECONDS=$((NOW_EPOCH - START_EPOCH))
            UPTIME_HOURS=$((UPTIME_SECONDS / 3600))
            UPTIME_MINUTES=$(((UPTIME_SECONDS % 3600) / 60))
            echo -e "${BLUE}● Uptime:${NC} ${UPTIME_HOURS}h ${UPTIME_MINUTES}m"
            
            # Estimate cost
            HOURLY_RATE=0.12  # Approximate for e2-highcpu-16 spot
            COST=$(echo "scale=2; $UPTIME_HOURS * $HOURLY_RATE" | bc)
            echo -e "${BLUE}● Estimated Cost:${NC} \$${COST} (at ~\$${HOURLY_RATE}/hour spot pricing)"
        fi
    fi
}

check_progress() {
    # Count completed iterations from GCS
    echo -e "\n${CYAN}Checking Progress...${NC}"
    
    # Get counts for each result type
    ZONE_COUNT=$(gsutil ls gs://$BUCKET_NAME/results/parquet_files/zone_simulation_result_*.parquet 2>/dev/null | wc -l)
    DISTRICT_COUNT=$(gsutil ls gs://$BUCKET_NAME/results/parquet_files/district_simulation_result_*.parquet 2>/dev/null | wc -l)
    COUNTRIES_COUNT=$(gsutil ls gs://$BUCKET_NAME/results/parquet_files/countries_simulation_result_*.parquet 2>/dev/null | wc -l)
    
    # Use minimum count as the true completion count
    COMPLETED=$ZONE_COUNT
    if [ $DISTRICT_COUNT -lt $COMPLETED ]; then
        COMPLETED=$DISTRICT_COUNT
    fi
    if [ $COUNTRIES_COUNT -lt $COMPLETED ]; then
        COMPLETED=$COUNTRIES_COUNT
    fi
    
    PERCENTAGE=$((COMPLETED * 100 / TOTAL_ITERATIONS))
    REMAINING=$((TOTAL_ITERATIONS - COMPLETED))
    
    # Progress bar
    BAR_LENGTH=40
    FILLED_LENGTH=$((COMPLETED * BAR_LENGTH / TOTAL_ITERATIONS))
    EMPTY_LENGTH=$((BAR_LENGTH - FILLED_LENGTH))
    
    echo -ne "${GREEN}Progress: ["
    for ((i=0; i<FILLED_LENGTH; i++)); do echo -n "█"; done
    for ((i=0; i<EMPTY_LENGTH; i++)); do echo -n "░"; done
    echo -e "] ${PERCENTAGE}%${NC}"
    
    echo -e "${MAGENTA}● Completed:${NC} $COMPLETED / $TOTAL_ITERATIONS iterations"
    echo -e "${MAGENTA}● Remaining:${NC} $REMAINING iterations"
    
    # Show individual counts if they differ
    if [ $ZONE_COUNT -ne $DISTRICT_COUNT ] || [ $ZONE_COUNT -ne $COUNTRIES_COUNT ]; then
        echo -e "${YELLOW}  Note: Result files are uneven:${NC}"
        echo -e "  - Zone results: $ZONE_COUNT"
        echo -e "  - District results: $DISTRICT_COUNT"
        echo -e "  - Countries results: $COUNTRIES_COUNT"
    fi
    
    # Estimate completion time
    if [ $COMPLETED -gt 10 ] && [ -n "$CREATED" ]; then
        START_EPOCH=$(date -d "$CREATED" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$CREATED" +%s)
        NOW_EPOCH=$(date +%s)
        ELAPSED=$((NOW_EPOCH - START_EPOCH))
        
        if [ $COMPLETED -gt 0 ]; then
            SECONDS_PER_ITERATION=$((ELAPSED / COMPLETED))
            REMAINING_SECONDS=$((SECONDS_PER_ITERATION * REMAINING))
            REMAINING_HOURS=$((REMAINING_SECONDS / 3600))
            REMAINING_MINUTES=$(((REMAINING_SECONDS % 3600) / 60))
            
            echo -e "${CYAN}● Est. Time Remaining:${NC} ${REMAINING_HOURS}h ${REMAINING_MINUTES}m"
            
            # Estimate completion time
            COMPLETION_EPOCH=$((NOW_EPOCH + REMAINING_SECONDS))
            COMPLETION_TIME=$(date -d "@$COMPLETION_EPOCH" '+%Y-%m-%d %H:%M' 2>/dev/null || date -r $COMPLETION_EPOCH '+%Y-%m-%d %H:%M')
            echo -e "${CYAN}● Est. Completion:${NC} $COMPLETION_TIME"
            
            # Estimate total cost
            TOTAL_HOURS=$((ELAPSED / 3600 + REMAINING_HOURS))
            HOURLY_RATE=0.12
            TOTAL_COST=$(echo "scale=2; $TOTAL_HOURS * $HOURLY_RATE" | bc)
            echo -e "${CYAN}● Est. Total Cost:${NC} \$${TOTAL_COST}"
        fi
    fi
    
    return $COMPLETED
}

check_checkpoint() {
    echo -e "\n${CYAN}Checking Checkpoint Status...${NC}"
    
    # Download and check checkpoint file
    TEMP_CHECKPOINT="/tmp/checkpoint_check.json"
    if gsutil cp gs://$BUCKET_NAME/results/parquet_files/checkpoint.json $TEMP_CHECKPOINT 2>/dev/null; then
        if command -v python3 &> /dev/null; then
            python3 << EOF
import json
from datetime import datetime

with open('$TEMP_CHECKPOINT', 'r') as f:
    checkpoint = json.load(f)

completed = len(checkpoint.get('completed_iterations', []))
total = checkpoint.get('total_iterations', 1000)
timestamp = checkpoint.get('timestamp', 0)

if timestamp:
    dt = datetime.fromtimestamp(timestamp)
    print(f"● Last Checkpoint: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"● Checkpoint Progress: {completed}/{total} iterations")
EOF
        else
            echo -e "${GREEN}● Checkpoint exists${NC}"
        fi
        rm $TEMP_CHECKPOINT
    else
        echo -e "${YELLOW}● No checkpoint found${NC}"
    fi
}

show_recent_logs() {
    echo -e "\n${CYAN}Recent Activity (from instance):${NC}"
    
    if [ "$STATUS" == "RUNNING" ]; then
        # Try to get recent log entries via SSH
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="tail -5 ~/simulation.log 2>/dev/null || echo 'Simulation log not available yet'" 2>/dev/null || echo "Unable to fetch logs"
    else
        echo "Instance not running - cannot fetch logs"
    fi
}

continuous_monitor() {
    while true; do
        clear
        print_header
        echo
        
        # Check instance
        if check_instance_status; then
            get_instance_details
        fi
        
        # Check progress
        check_progress
        LAST_COUNT=$?
        
        # Check checkpoint
        check_checkpoint
        
        # Show logs if running
        if [ "$STATUS" == "RUNNING" ]; then
            show_recent_logs
        fi
        
        echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}Refreshing in 60 seconds... (Press Ctrl+C to exit)${NC}"
        
        # Check if simulation is complete
        if [ $LAST_COUNT -ge $TOTAL_ITERATIONS ]; then
            echo -e "\n${GREEN}✅ SIMULATION COMPLETE!${NC}"
            echo -e "${GREEN}Run './gcp/deploy-spot.sh download' to get results${NC}"
            break
        fi
        
        sleep 60
    done
}

quick_check() {
    print_header
    echo
    
    # Check instance
    if check_instance_status; then
        get_instance_details
    fi
    
    # Check progress
    check_progress
    COMPLETED=$?
    
    # Check checkpoint
    check_checkpoint
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    
    # Summary
    if [ $COMPLETED -ge $TOTAL_ITERATIONS ]; then
        echo -e "${GREEN}✅ SIMULATION COMPLETE!${NC}"
        echo -e "${GREEN}Run './gcp/deploy-spot.sh download' to get results${NC}"
    elif [ "$STATUS" == "RUNNING" ]; then
        echo -e "${GREEN}Simulation is running. Use --watch for continuous monitoring.${NC}"
    elif [ "$STATUS" == "TERMINATED" ] || [ "$STATUS" == "STOPPED" ]; then
        echo -e "${YELLOW}Instance was preempted or stopped. Run './gcp/deploy-spot.sh restart' to resume.${NC}"
    else
        echo -e "${RED}Instance not found. Run './gcp/deploy-spot.sh deploy' to start.${NC}"
    fi
}

download_partial() {
    echo -e "${CYAN}Downloading partial results...${NC}"
    
    mkdir -p ./results/parquet_files
    
    # Download all available results
    gsutil -m rsync -r gs://$BUCKET_NAME/results/ ./results/
    
    # Count local results
    LOCAL_COUNT=$(ls ./results/parquet_files/*_simulation_result_*.parquet 2>/dev/null | wc -l)
    echo -e "${GREEN}Downloaded $LOCAL_COUNT result files${NC}"
    
    # Check if we can process partial results
    if [ $LOCAL_COUNT -gt 0 ]; then
        echo -e "${CYAN}You can process these partial results using:${NC}"
        echo "  python scripts/process_partial_results.py"
    fi
}

# Main script
case "${1:-}" in
    --watch|-w)
        continuous_monitor
        ;;
    --download|-d)
        quick_check
        download_partial
        ;;
    *)
        quick_check
        ;;
esac
#!/bin/bash

# Water Access Monte Carlo - GCP Spot VM Deployment Script
# This script automates the deployment and management of the simulation on GCP

set -e

# Configuration
PROJECT_ID="water-access-compute"
ZONE="us-central1-a"
INSTANCE_NAME="water-access-compute-spot"
MACHINE_TYPE="e2-highcpu-16"  # Change to e2-highcpu-32 for faster completion
BUCKET_NAME="water-access-data"
MAX_RUN_DURATION="24h"
BOOT_DISK_SIZE="100GB"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated. Run: gcloud auth login"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Check if APIs are enabled
    print_status "Checking required APIs..."
    gcloud services enable compute.googleapis.com storage-api.googleapis.com
    
    print_status "Prerequisites check completed"
}

create_bucket() {
    print_status "Setting up storage bucket..."
    
    # Check if bucket exists
    if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
        print_status "Bucket gs://$BUCKET_NAME already exists"
    else
        print_status "Creating bucket gs://$BUCKET_NAME..."
        gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://$BUCKET_NAME
    fi
    
    # Create directories
    gsutil -m mkdir -p gs://$BUCKET_NAME/input/
    gsutil -m mkdir -p gs://$BUCKET_NAME/results/
    gsutil -m mkdir -p gs://$BUCKET_NAME/checkpoints/
    gsutil -m mkdir -p gs://$BUCKET_NAME/code/
    
    print_status "Storage bucket ready"
}

upload_data() {
    print_status "Uploading code and data to GCS..."
    
    # Check if we're in the right directory
    if [ ! -f "environment.yml" ]; then
        print_error "Please run this script from the water-access project root directory"
        exit 1
    fi
    
    # Upload code (always update)
    print_status "Uploading code files..."
    gsutil -m -q cp -r src scripts environment.yml setup.py gs://$BUCKET_NAME/code/
    
    # Check if data already uploaded
    if gsutil ls gs://$BUCKET_NAME/input/data/ &> /dev/null; then
        print_warning "Data already uploaded. Skipping data upload (3.5GB)..."
        print_warning "To re-upload data, run: gsutil -m rm -r gs://$BUCKET_NAME/input/data"
    else
        print_status "Uploading data (3.5GB) - this will take a while..."
        gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp -r data gs://$BUCKET_NAME/input/
    fi
    
    print_status "Upload completed"
}

create_startup_script() {
    print_status "Creating startup script..."
    
    cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
set -e

# Log startup
echo "Starting water-access simulation setup at $(date)" | tee /startup.log

# Install system dependencies
apt-get update
apt-get install -y wget bzip2 git build-essential

# Install Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/bin/activate

# Install mamba for faster conda operations
conda install -n base -c conda-forge mamba -y

# Create work directory
mkdir -p $HOME/water-access
cd $HOME/water-access

# Download code from GCS
echo "Downloading code from GCS..." | tee -a /startup.log
gsutil -m cp -r gs://water-access-data/code/* .

# Download data from GCS
echo "Downloading data from GCS..." | tee -a /startup.log
gsutil -m cp -r gs://water-access-data/input/data .

# Setup Python environment
echo "Setting up Python environment..." | tee -a /startup.log
mamba env create -f environment.yml -y || mamba env update -f environment.yml
source activate water-access
pip install -e .

# Create results directory
mkdir -p results/parquet_files

# Check for existing checkpoint and results
echo "Checking for existing checkpoints..." | tee -a /startup.log
gsutil -m cp -r gs://water-access-data/checkpoints/* results/parquet_files/ 2>/dev/null || true
gsutil -m cp -r gs://water-access-data/results/parquet_files/*.parquet results/parquet_files/ 2>/dev/null || true

# Count existing results
EXISTING_COUNT=$(ls results/parquet_files/*_simulation_result_*.parquet 2>/dev/null | wc -l)
echo "Found $EXISTING_COUNT existing simulation results" | tee -a /startup.log

# Setup automatic result sync every 30 minutes
(crontab -l 2>/dev/null || true; echo "*/30 * * * * cd $HOME/water-access && gsutil -m -q rsync -r results/ gs://water-access-data/results/") | crontab -

# Run simulation
echo "Starting Monte Carlo simulation at $(date)" | tee -a /startup.log
cd $HOME/water-access
nohup python scripts/run_monte_carlo_gcp.py > simulation.log 2>&1 &

echo "Simulation started. PID: $(pgrep -f run_monte_carlo_gcp.py)" | tee -a /startup.log
echo "Setup completed at $(date)" | tee -a /startup.log
EOF
    
    print_status "Startup script created"
}

create_instance() {
    print_status "Creating Spot VM instance..."
    
    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
        print_warning "Instance $INSTANCE_NAME already exists"
        read -p "Delete and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Deleting existing instance..."
            gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        else
            print_status "Using existing instance"
            return
        fi
    fi
    
    # Create startup script
    create_startup_script
    
    # Create instance
    print_status "Creating instance $INSTANCE_NAME..."
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --max-run-duration=$MAX_RUN_DURATION \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-standard \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --scopes=storage-rw \
        --metadata-from-file=startup-script=/tmp/startup-script.sh
    
    print_status "Instance created successfully"
    print_status "Machine type: $MACHINE_TYPE"
    print_status "Spot pricing: ~\$0.12/hour (70-90% cheaper than on-demand)"
    
    # Clean up
    rm /tmp/startup-script.sh
}

monitor_instance() {
    print_status "Monitoring instance startup..."
    
    # Wait for instance to be running
    print_status "Waiting for instance to be ready..."
    sleep 30
    
    # Show serial port output
    print_status "Checking startup logs..."
    gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE | tail -20
    
    print_status "Instance is ready. Connecting..."
    echo
    echo "=========================================="
    echo "To monitor progress, SSH into the instance:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo "Then run:"
    echo "  tail -f ~/simulation.log"
    echo "  ls ~/water-access/results/parquet_files/*.parquet | wc -l"
    echo "=========================================="
}

check_progress() {
    print_status "Checking simulation progress..."
    
    # Check if instance exists
    if ! gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
        print_error "Instance $INSTANCE_NAME not found"
        return
    fi
    
    # Get progress from GCS
    TOTAL_ITERATIONS=1000
    COMPLETED=$(gsutil ls gs://$BUCKET_NAME/results/parquet_files/zone_simulation_result_*.parquet 2>/dev/null | wc -l)
    
    if [ $COMPLETED -gt 0 ]; then
        PERCENTAGE=$((COMPLETED * 100 / TOTAL_ITERATIONS))
        print_status "Progress: $COMPLETED/$TOTAL_ITERATIONS iterations ($PERCENTAGE%)"
        
        # Estimate time remaining (rough estimate)
        if [ $COMPLETED -gt 10 ]; then
            INSTANCE_START=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(creationTimestamp)")
            START_EPOCH=$(date -d "$INSTANCE_START" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$INSTANCE_START" +%s)
            NOW_EPOCH=$(date +%s)
            ELAPSED=$((NOW_EPOCH - START_EPOCH))
            RATE=$(echo "scale=2; $ELAPSED / $COMPLETED" | bc)
            REMAINING=$((TOTAL_ITERATIONS - COMPLETED))
            ETA=$(echo "scale=0; $RATE * $REMAINING / 3600" | bc)
            print_status "Estimated time remaining: ${ETA} hours"
        fi
    else
        print_warning "No results found yet. Simulation may still be starting..."
    fi
}

download_results() {
    print_status "Downloading results..."
    
    # Create local results directory
    mkdir -p ./results
    
    # Download results
    gsutil -m rsync -r gs://$BUCKET_NAME/results/ ./results/
    
    # Count results
    RESULTS_COUNT=$(ls ./results/parquet_files/*_simulation_result_*.parquet 2>/dev/null | wc -l)
    print_status "Downloaded $RESULTS_COUNT result files"
}

stop_instance() {
    print_status "Stopping instance..."
    
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
        gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
        print_status "Instance stopped (can be restarted later)"
    else
        print_warning "Instance not found"
    fi
}

delete_instance() {
    print_status "Deleting instance..."
    
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        print_status "Instance deleted"
    else
        print_warning "Instance not found"
    fi
}

restart_instance() {
    print_status "Restarting instance..."
    
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
        gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
        print_status "Instance restarted"
        monitor_instance
    else
        print_error "Instance not found. Run 'deploy' first."
    fi
}

# Main script
case "${1:-}" in
    deploy)
        print_status "Starting full deployment..."
        check_prerequisites
        create_bucket
        upload_data
        create_instance
        monitor_instance
        ;;
    
    upload)
        print_status "Uploading code and data..."
        check_prerequisites
        create_bucket
        upload_data
        ;;
    
    create)
        print_status "Creating instance..."
        check_prerequisites
        create_instance
        monitor_instance
        ;;
    
    progress)
        check_progress
        ;;
    
    download)
        download_results
        ;;
    
    stop)
        stop_instance
        ;;
    
    restart)
        restart_instance
        ;;
    
    delete)
        delete_instance
        ;;
    
    ssh)
        print_status "Connecting to instance..."
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
        ;;
    
    logs)
        print_status "Fetching logs..."
        gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE | tail -100
        ;;
    
    *)
        echo "Water Access Monte Carlo - GCP Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Full deployment (upload data, create instance, start simulation)"
        echo "  upload    - Upload code and data to GCS only"
        echo "  create    - Create and start Spot VM instance"
        echo "  progress  - Check simulation progress"
        echo "  download  - Download results from GCS"
        echo "  stop      - Stop instance (keeps disk, can restart)"
        echo "  restart   - Restart stopped instance"
        echo "  delete    - Delete instance (removes everything)"
        echo "  ssh       - SSH into running instance"
        echo "  logs      - View instance startup logs"
        echo ""
        echo "Typical workflow:"
        echo "  1. $0 deploy       # Initial setup and start"
        echo "  2. $0 progress     # Check progress periodically"
        echo "  3. $0 download     # Get results"
        echo "  4. $0 delete       # Clean up when done"
        echo ""
        echo "If interrupted:"
        echo "  $0 restart         # Resume from checkpoint"
        ;;
esac
#!/bin/bash

# Quick test deployment for GCP - minimal cost testing
# This script tests the GCP setup with only 2 iterations

set -e

# Configuration
PROJECT_ID="water-access-compute"
ZONE="us-central1-a"
INSTANCE_NAME="water-access-test"
MACHINE_TYPE="e2-micro"  # Smallest instance for testing
BUCKET_NAME="water-access-data"
MAX_RUN_DURATION="1h"  # Auto-terminate after 1 hour for safety
BOOT_DISK_SIZE="20GB"  # Smaller disk for test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}       GCP TEST DEPLOYMENT - MINIMAL COST TESTING${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

print_status "This will create a small test VM to verify GCP setup"
print_status "Estimated cost: \$0.01-0.02 (less than 5 cents)"
print_status "Auto-terminates after 1 hour for safety"
echo ""

read -p "Continue with test deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Test cancelled"
    exit 0
fi

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install it first."
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Check if test instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    print_warning "Test instance already exists. Deleting it first..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
fi

# Create minimal startup script for test
print_status "Creating test startup script..."

cat > /tmp/test-startup.sh << 'EOF'
#!/bin/bash
echo "TEST DEPLOYMENT - Starting at $(date)" | tee /test.log

# Install minimal dependencies
apt-get update
apt-get install -y wget python3-pip

# Create test marker file
echo "TEST_RUN" > /home/test_marker.txt

# Download test script from GCS (if exists)
mkdir -p /home/water-access-test
cd /home/water-access-test

# Simple Python test
cat > test.py << 'PYTEST'
import time
print("GCP Test: Python is working!")
print("Starting test simulation...")
time.sleep(10)
print("Test completed successfully!")
with open("/home/test_complete.txt", "w") as f:
    f.write("Test completed at: " + str(time.time()))
PYTEST

python3 test.py | tee -a /test.log

echo "TEST DEPLOYMENT - Completed at $(date)" | tee -a /test.log
EOF

# Create test instance
print_status "Creating test instance ($MACHINE_TYPE)..."

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --max-run-duration=$MAX_RUN_DURATION \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata-from-file=startup-script=/tmp/test-startup.sh \
    --labels=purpose=test,auto-delete=yes

print_status "Test instance created!"

# Clean up temp file
rm /tmp/test-startup.sh

# Wait a bit for instance to start
print_status "Waiting for instance to initialize..."
sleep 30

# Check instance status
print_status "Checking instance status..."
gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="table(
    name,
    status,
    machineType.scope(machineTypes),
    scheduling.provisioningModel,
    scheduling.instanceTerminationAction
)"

# Get serial output
print_status "Checking startup logs..."
gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE | tail -20

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ TEST DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "Test instance '$INSTANCE_NAME' is running"
echo "It will auto-terminate in 1 hour (or you can delete it now)"
echo ""
echo "To check the test:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='cat /test.log'"
echo ""
echo "To run the actual Monte Carlo test (2 iterations):"
echo "  1. Upload the test script:"
echo "     gcloud compute scp scripts/run_monte_carlo_test.py $INSTANCE_NAME:~/ --zone=$ZONE"
echo "  2. SSH and run it:"
echo "     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the test instance (stop charges):"
echo -e "  ${YELLOW}gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE${NC}"
echo ""
echo "Estimated cost so far: < \$0.01"
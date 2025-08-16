# Google Cloud Setup for Water Access Monte Carlo Simulations

## Overview
This guide sets up a cost-optimized GCP environment for running compute-intensive Monte Carlo simulations using Spot VMs, which can save 70-90% on compute costs.

## Prerequisites

1. **Google Cloud Account** with billing enabled (you get $300 free credits)
2. **gcloud CLI** installed and authenticated:
```bash
# Check authentication
gcloud auth list

# Set project
gcloud config set project water-access-compute
```

3. **Git LFS** for handling large data files:
```bash
git lfs pull  # Ensure all data files are downloaded
```

## Cost Estimates

| Configuration | vCPUs | Memory | Spot Price | Est. Runtime | Total Cost |
|--------------|-------|---------|------------|--------------|------------|
| e2-highcpu-16 (Spot) | 16 | 16 GB | ~$0.12/hr | 4-5 days | $12-15 |
| e2-highcpu-32 (Spot) | 32 | 32 GB | ~$0.24/hr | 2-3 days | $12-18 |
| n2-highcpu-32 (Spot) | 32 | 32 GB | ~$0.36/hr | 2 days | $18-25 |
| n2-highcpu-64 (Spot) | 64 | 64 GB | ~$0.72/hr | 1 day | $18-25 |

**Recommended**: `e2-highcpu-16` or `e2-highcpu-32` for best cost/performance ratio

## Step 1: Enable Required APIs

```bash
# Enable necessary APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable cloudbilling.googleapis.com
```

## Step 2: Create Storage Bucket

```bash
# Create bucket for data and results (one-time setup)
gsutil mb -p water-access-compute -c STANDARD -l us-central1 gs://water-access-data

# Create folders
gsutil -m mkdir gs://water-access-data/input/
gsutil -m mkdir gs://water-access-data/results/
gsutil -m mkdir gs://water-access-data/checkpoints/
```

## Step 3: Upload Data to Cloud Storage

```bash
# Upload data folder (3.5GB) - do this once
cd /path/to/water-access
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp -r data gs://water-access-data/input/

# Upload code
gsutil -m cp -r src scripts environment.yml gs://water-access-data/code/
```

## Step 4: Create Spot VM Instance

### Option A: Basic e2-highcpu-16 Spot VM (Recommended for cost)
```bash
gcloud compute instances create water-access-compute-spot \
    --zone=us-central1-a \
    --machine-type=e2-highcpu-16 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration=24h \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=storage-rw \
    --metadata=startup-script='#!/bin/bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/bin/activate

# Download code and data from GCS
gsutil -m cp -r gs://water-access-data/code/* $HOME/water-access/
gsutil -m cp -r gs://water-access-data/input/data $HOME/water-access/

# Setup environment
cd $HOME/water-access
mamba env create -f environment.yml
mamba activate water-access
pip install -e .

# Create results directory
mkdir -p results/parquet_files

# Check for existing checkpoint
gsutil -m cp gs://water-access-data/checkpoints/* results/parquet_files/ 2>/dev/null || true

# Run simulation with checkpointing
nohup python scripts/run_monte_carlo_gcp.py > simulation.log 2>&1 &

# Setup periodic checkpoint upload (every hour)
(crontab -l 2>/dev/null; echo "0 * * * * gsutil -m rsync -r $HOME/water-access/results/ gs://water-access-data/results/") | crontab -
'
```

### Option B: More powerful e2-highcpu-32 Spot VM (Faster completion)
```bash
gcloud compute instances create water-access-compute-spot \
    --zone=us-central1-a \
    --machine-type=e2-highcpu-32 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration=24h \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=storage-rw \
    --metadata-from-file=startup-script=gcp/startup-script.sh
```

## Step 5: Connect to VM and Monitor Progress

```bash
# SSH into the instance
gcloud compute ssh water-access-compute-spot --zone=us-central1-a

# Once connected, monitor progress
tail -f ~/simulation.log

# Check completed iterations
ls ~/water-access/results/parquet_files/*.parquet | wc -l

# Check resource usage
htop
```

## Step 6: Handle Spot VM Preemptions

Spot VMs can be preempted with 30-second notice. The `run_monte_carlo_gcp.py` script handles this automatically by:
1. Saving progress every 10 iterations
2. Detecting completed iterations on restart
3. Resuming from last checkpoint

### If VM is preempted, restart it:
```bash
# Check instance status
gcloud compute instances list

# If stopped, restart
gcloud compute instances start water-access-compute-spot --zone=us-central1-a

# SSH back in and check progress
gcloud compute ssh water-access-compute-spot --zone=us-central1-a
cd ~/water-access
tail -f simulation.log
```

## Step 7: Download Results

```bash
# Download results to local machine
gsutil -m cp -r gs://water-access-data/results/* ./results/

# Or sync periodically
gsutil -m rsync -r gs://water-access-data/results/ ./results/
```

## Step 8: Clean Up (Important!)

```bash
# Delete the VM instance
gcloud compute instances delete water-access-compute-spot --zone=us-central1-a

# Optional: Delete storage bucket if no longer needed
# gsutil -m rm -r gs://water-access-data
```

## Monitoring and Debugging

### Check VM logs
```bash
gcloud compute instances get-serial-port-output water-access-compute-spot --zone=us-central1-a
```

### Check simulation progress
```bash
# SSH into VM
gcloud compute ssh water-access-compute-spot --zone=us-central1-a

# Check log
tail -f ~/simulation.log

# Count completed iterations
ls ~/water-access/results/parquet_files/*_simulation_result_*.parquet | wc -l

# Check checkpoint
cat ~/water-access/results/parquet_files/checkpoint.json
```

### Check costs
```bash
# View current month's forecast
gcloud billing accounts list
gcloud alpha billing accounts budgets list --billing-account=YOUR_BILLING_ACCOUNT_ID
```

## Tips for Cost Optimization

1. **Use Spot VMs**: 70-90% cheaper than regular VMs
2. **Choose e2 machine types**: Better price/performance for this workload
3. **Use us-central1**: Generally cheapest region
4. **Set max-run-duration**: Prevents runaway costs
5. **Delete instances promptly**: Stop paying as soon as job completes
6. **Use Standard storage**: Cheaper than SSD for this use case
7. **Monitor actively**: Check progress regularly to avoid waste

## Troubleshooting

### If simulation doesn't start:
```bash
# Check startup script output
gcloud compute instances get-serial-port-output water-access-compute-spot --zone=us-central1-a

# SSH in and run manually
gcloud compute ssh water-access-compute-spot --zone=us-central1-a
cd ~/water-access
source ~/miniconda3/bin/activate
mamba activate water-access
python scripts/run_monte_carlo_gcp.py
```

### If preempted frequently:
- Try a different zone (us-central1-b, us-central1-c)
- Consider using a slightly more expensive machine type
- Run during off-peak hours (nights/weekends)

### If running out of disk space:
```bash
# Resize disk (while VM is stopped)
gcloud compute disks resize water-access-compute-spot-boot --size=200GB --zone=us-central1-a
```

## For Your Colleague

To run the same simulation:
1. Get added to the GCP project (ask project owner)
2. Install gcloud CLI and authenticate
3. Follow steps 4-8 (data is already uploaded)
4. Results will sync to the shared bucket

The checkpoint system ensures you won't duplicate work - the script automatically detects and skips completed iterations.
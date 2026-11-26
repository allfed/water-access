# Quick GCP Test - Minimal Cost Verification

## Overview
This is a minimal test to verify your GCP setup works correctly with very low cost (< $0.05).

## Test Options

### Option 1: Super Quick Test (2 minutes, ~$0.01)
Just verify GCP authentication and VM creation work:

```bash
# Run basic VM test
./gcp/test-deploy.sh

# Check it worked (should show test logs)
gcloud compute ssh water-access-test --zone=us-central1-a --command='cat /test.log'

# Delete immediately to stop charges
gcloud compute instances delete water-access-test --zone=us-central1-a
```

### Option 2: Full Test with 2 Monte Carlo Iterations (~1 hour, ~$0.15)
Test the actual simulation pipeline with 2 iterations on the sample grid
(`data/GIS/GIS_data_zones_sample.csv`, a few thousand cells):

```bash
# 1. Upload code and data (also used by the production run)
./gcp/deploy-spot.sh upload

# 2. Create VM with simulation (most of the time goes on building the environment)
gcloud compute instances create water-access-test \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --provisioning-model=SPOT \
    --max-run-duration=2h \
    --boot-disk-size=30GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=storage-rw \
    --metadata=startup-script='#!/bin/bash
mkdir -p /home/test && cd /home/test
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
bash miniforge.sh -b -p /home/test/miniforge3
source /home/test/miniforge3/etc/profile.d/conda.sh
gsutil -m cp -r gs://water-access-james-data/code/* .
gsutil -m cp -r gs://water-access-james-data/input/data .
mamba env create -f environment.yml -y && conda activate water-access && pip install -e .
python scripts/run_monte_carlo.py --input data/GIS/GIS_data_zones_sample.csv \
    --iterations 2 --output-dir results/test 2>&1 | tee test_output.log
'

# 3. Monitor progress
# Wait 5-10 minutes, then check
gcloud compute ssh water-access-test --zone=us-central1-a --command='tail -20 /home/test/test_output.log'

# 4. Clean up when done
gcloud compute instances delete water-access-test --zone=us-central1-a
```

## What to Expect

### Successful Output:
```
TEST DEPLOYMENT - Starting at [timestamp]
GCP Test: Python is working!
Starting test simulation...
Test completed successfully!
TEST DEPLOYMENT - Completed at [timestamp]
```

### For Monte Carlo test:
```
Zone frame: data/GIS/GIS_data_zones_sample.csv
...
Simulating: 100%|██████████| 2/2
...
Time taken: X.XX minutes
```
The summary CSVs are written to `results/test/`.

## If Something Goes Wrong

### Authentication Issues:
```bash
gcloud auth login
gcloud config set project water-access-compute
```

### Permission Issues:
```bash
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
```

### Instance Won't Start:
- Try a different zone: `--zone=us-central1-b`
- Check quotas in GCP Console

## Cost Breakdown
- **Option 1**: e2-micro for 2 minutes ≈ $0.008
- **Option 2**: e2-standard-4 Spot for about an hour ≈ $0.15
- **Storage**: Negligible for test files

**Total**: Less than $0.20 for complete testing

## Next Steps
Once test works:
1. Run the full production deployment: `./gcp/deploy-spot.sh deploy`
2. Monitor with: `./gcp/monitor.sh --watch`
3. Cost for full run: $8-17 for 1000 draws on Spot VMs
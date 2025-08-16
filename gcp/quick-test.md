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

### Option 2: Full Test with 2 Monte Carlo Iterations (~30 min, ~$0.03)
Test the actual simulation pipeline with 2 iterations:

```bash
# 1. Upload minimal test data first
gsutil -m mkdir -p gs://water-access-data/test/
gsutil -m cp scripts/run_monte_carlo_test.py gs://water-access-data/test/

# 2. Create VM with simulation
gcloud compute instances create water-access-test \
    --zone=us-central1-a \
    --machine-type=e2-small \
    --provisioning-model=SPOT \
    --max-run-duration=1h \
    --boot-disk-size=30GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=storage-rw \
    --metadata=startup-script='#!/bin/bash
# Quick test setup
apt-get update && apt-get install -y python3-pip
pip3 install pandas numpy tqdm
mkdir -p /home/test
cd /home/test
gsutil cp gs://water-access-data/test/run_monte_carlo_test.py .
python3 run_monte_carlo_test.py | tee test_output.log
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
RUNNING TEST VERSION - ONLY 2 ITERATIONS
This is for testing GCP setup, not production runs
...
✅ Saved test results for iteration 0
✅ Saved test results for iteration 1
TEST COMPLETE!
Time taken: X.XX minutes
✅ Test simulation successful! GCP setup is working correctly.
```

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
- **Option 2**: e2-small for 30 minutes ≈ $0.024
- **Storage**: Negligible for test files

**Total**: Less than $0.05 for complete testing

## Next Steps
Once test works:
1. Run the full production deployment: `./gcp/deploy-spot.sh deploy`
2. Monitor with: `./gcp/monitor.sh --watch`
3. Cost for full run: $12-20 instead of $200-400
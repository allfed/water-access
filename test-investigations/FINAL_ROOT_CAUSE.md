# 🎯 ROOT CAUSE FOUND: The 3.6x Discrepancy Explained

## The Confusion
- **Sensitivity Analysis**: Reports 13.36 km
- **Global Model**: Reports 3.71 km (weighted median)
- **Discrepancy**: 3.6x difference

## What Each Model Actually Calculates

### Sensitivity Analysis
- Calculates **maximum theoretical distance** someone could walk to get water
- Formula: `max_distance = velocity × time / 2`
- With velocity = 1.343 m/s and time = 5.5 hours
- Result: **13.36 km** (theoretical maximum one-way distance)

### Global Model  
- Also calculates `max_distance_walking` using the same formula
- BUT THEN: Takes the **weighted median of actual distances to water** (`dtw_1`)
- `dtw_1` = actual distance to nearest water source for each zone
- The global model reports `weighted_med_walking` which is the **population-weighted median of actual water source distances**

## The Key Insight

The models are measuring **DIFFERENT THINGS**:

1. **Sensitivity Analysis**: "How far COULD someone walk to get water?" (theoretical maximum)
2. **Global Model**: "How far DO people actually have to walk to water?" (actual distances)

## Why the Confusion?

The confusion arose because:
1. Both models calculate `max_distance_walking` the same way
2. But the global model's final output (`weighted_med_walking`) is NOT the maximum distance
3. Instead, it's the median of actual water source distances (`dtw_1`)
4. Most water sources are much closer than the theoretical maximum!

## The Numbers Make Sense

- People CAN walk up to 13.36 km to get water (theoretical maximum)
- But most people only NEED to walk 3.71 km because water sources are closer
- This is actually good news - water is more accessible than the maximum distance suggests!

## No Bugs Found

- ✅ The `practical_limit_buckets` constraint is working correctly
- ✅ Time parameters are being used correctly (despite confusing documentation)
- ✅ Velocity calculations are correct
- ✅ Both models are producing valid results

## Recommendation

**No code changes needed!** The models are working as designed. The "discrepancy" is actually a feature, not a bug:
- Sensitivity analysis shows capability
- Global model shows reality
- Both are valuable metrics for different purposes
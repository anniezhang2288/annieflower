# Getting Started: Your First Performance Model

This guide will get you up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd annieflower
pip install -r requirements.txt
```

## Step 2: Run the Walkthrough

The easiest way to see everything in action:

```bash
python walkthrough.py
```

This will:
- Create a test trace
- Configure microarchitecture
- Initialize all models
- Process the trace
- Show you the results
- Compare different configurations

## Step 3: Try Your Own Trace

### Option A: Create a Simple Test Trace

```python
# my_first_trace.py
import json

trace = [
    {"pc": 0x1000, "instruction_type": "load", "memory_address": 0x2000},
    {"pc": 0x1004, "instruction_type": "alu"},
    {"pc": 0x1008, "instruction_type": "store", "memory_address": 0x2004},
    {"pc": 0x100c, "instruction_type": "branch", "branch_taken": True},
]

with open("traces/my_trace.json", "w") as f:
    json.dump(trace, f, indent=2)

print("Created trace: traces/my_trace.json")
```

### Option B: Use Your Tacit Trace

If you have a Tacit trace file, just place it in the `traces/` directory.

## Step 4: Run Your First Prediction

```python
# my_first_prediction.py
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework

# Configure CPU
config = MicroarchConfig(
    pipeline_width=4,
    rob_size=192,
    l1d_size=32 * 1024,
    l2_size=256 * 1024,
    l3_size=8 * 1024 * 1024
)

# Initialize framework
framework = PerformanceModelFramework(config)

# Predict performance
results = framework.process_trace_file("traces/my_trace.json")

# Print results
print(f"Predicted IPC: {results['performance_breakdown']['ipc']:.3f}")
print(f"Total Cycles: {results['performance_breakdown']['total_cycles']:.0f}")
```

Run it:
```bash
python my_first_prediction.py
```

## What You Just Did

1. ✅ Created/loaded an instruction trace
2. ✅ Configured a CPU microarchitecture
3. ✅ Initialized performance models
4. ✅ Predicted performance (IPC, cycles, stalls)
5. ✅ Got component-level breakdown

## Next Steps

### Learn More
- **TUTORIAL.md**: Complete detailed tutorial with explanations
- **SETUP_GUIDE.md**: Detailed setup instructions
- **ARCHITECTURE.md**: How everything works under the hood
- **QUICK_REFERENCE.md**: Quick lookup for common operations

### Explore
- Modify microarchitectural parameters
- Try different traces
- Compare configurations
- Analyze bottlenecks

### Extend
- Add new component models
- Integrate machine learning
- Calibrate against real hardware
- Add visualization

## Common Questions

**Q: How do I get real instruction traces?**  
A: Use Tacit, gem5, Pin, or other trace collection tools. See TUTORIAL.md Section 1.

**Q: How accurate are the predictions?**  
A: The models use analytical approximations. Calibrate them against real hardware for better accuracy.

**Q: Can I add my own models?**  
A: Yes! Inherit from `BasePerformanceModel` and add to the framework. See TUTORIAL.md Section 4.3.

**Q: How do I improve accuracy?**  
A: 1) Calibrate model parameters, 2) Add ML correction (like Concorde), 3) Use more detailed traces.

## File Overview

| File | Purpose |
|------|---------|
| `walkthrough.py` | Interactive demonstration |
| `TUTORIAL.md` | Complete step-by-step tutorial |
| `QUICK_REFERENCE.md` | Quick lookup guide |
| `SETUP_GUIDE.md` | Detailed setup instructions |
| `ARCHITECTURE.md` | Architecture explanation |
| `example_usage.py` | Code examples |

## Need Help?

1. Run `python walkthrough.py` to see everything in action
2. Read `TUTORIAL.md` for detailed explanations
3. Check `QUICK_REFERENCE.md` for quick answers
4. Look at `example_usage.py` for code examples

Happy modeling! 🚀


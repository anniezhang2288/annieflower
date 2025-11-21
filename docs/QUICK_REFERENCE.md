# Quick Reference Guide

## Getting Instruction Traces

### Using Tacit
```bash
tacit --program ./my_program --output trace.json
```

### Creating Test Trace (Python)
```python
import json
trace = [
    {"pc": 0x1000, "instruction_type": "load", "memory_address": 0x2000},
    {"pc": 0x1004, "instruction_type": "alu"},
]
with open("trace.json", "w") as f:
    json.dump(trace, f)
```

## Basic Usage

### 1. Configure Microarchitecture
```python
from analytical import MicroarchConfig

config = MicroarchConfig(
    pipeline_width=4,
    rob_size=192,
    l1d_size=32 * 1024,
    l2_size=256 * 1024,
    l3_size=8 * 1024 * 1024
)
```

### 2. Initialize Framework
```python
from performance_model import PerformanceModelFramework

framework = PerformanceModelFramework(config)
```

### 3. Process Trace
```python
results = framework.process_trace_file("traces/trace.json")
ipc = results['performance_breakdown']['ipc']
```

## Trace Format

### Required Fields
- `pc`: Program counter (instruction address)
- `instruction_type`: Type (load, store, branch, alu, etc.)

### Optional Fields
- `memory_address`: For load/store
- `branch_taken`: For branches
- `dependencies`: List of dependent instruction indices
- `cycle`: Cycle number

## Model Components

| Model | What It Models | Key Metrics |
|-------|---------------|-------------|
| **BP** | Branch predictor | Accuracy, misprediction stalls |
| **Cache** | Cache hierarchy | Hit rates, memory accesses |
| **ROB** | Reorder buffer | Occupancy, capacity stalls |
| **LSQ** | Load-store queue | Dependencies, capacity stalls |
| **Memory** | Memory system | Bandwidth, contention |

## Common Operations

### Compare Configurations
```python
configs = [config1, config2, config3]
comparison = framework.compare_configurations("trace.json", configs)
```

### Bottleneck Analysis
```python
analysis = framework.get_bottleneck_analysis("trace.json")
print(analysis['primary_bottleneck']['component'])
```

### Process Trace Directly
```python
trace_data = [{"pc": 0x1000, "instruction_type": "load"}]
results = framework.process_trace(trace_data)
```

### Save Results
```python
results = framework.predict_performance("trace.json", "results.json")
```

## Microarchitectural Parameters

### Pipeline
- `pipeline_width`: Instructions per cycle (default: 4)
- `pipeline_depth`: Pipeline stages (default: 14)

### Caches
- `l1d_size`: L1 data cache size (default: 32KB)
- `l1d_assoc`: L1D associativity (default: 8)
- `l2_size`: L2 cache size (default: 256KB)
- `l3_size`: L3 cache size (default: 8MB)

### Buffers
- `rob_size`: Reorder buffer size (default: 192)
- `lsq_size`: Load-store queue size (default: 72)
- `bp_size`: Branch predictor size (default: 4096)

### Memory
- `memory_latency`: Memory access latency in cycles (default: 100)

## Model Methods

### BasePerformanceModel
```python
process_trace(trace_data) -> PerformanceMetrics
estimate_latency(trace_data) -> float
estimate_throughput(trace_data) -> float
get_stall_cycles(trace_data) -> float
get_utilization(trace_data) -> float
```

### PerformanceModelFramework
```python
process_trace_file(trace_file) -> dict
process_trace(trace_data) -> dict
predict_performance(trace_file, output_file) -> dict
compare_configurations(trace_file, configs) -> dict
get_bottleneck_analysis(trace_file) -> dict
```

## PerformanceMetrics Structure

```python
PerformanceMetrics(
    latency: float,           # Total latency in cycles
    throughput: float,        # Throughput (0.0 to 1.0)
    utilization: float,      # Utilization (0.0 to 1.0)
    stall_cycles: float,      # Stall cycles
    additional_metrics: dict  # Component-specific metrics
)
```

## Results Structure

```python
{
    'performance_breakdown': {
        'total_instructions': int,
        'total_cycles': float,
        'total_stall_cycles': float,
        'ipc': float,
        'overall_throughput': float,
        'stall_breakdown': {component: stall_cycles}
    },
    'model_results': {
        'ComponentModel': {
            'latency': float,
            'throughput': float,
            'utilization': float,
            'stall_cycles': float,
            'additional_metrics': dict
        }
    },
    'trace_statistics': {instruction_type: count}
}
```

## Extending the Framework

### Add New Model
```python
from analytical.base_model import BasePerformanceModel

class MyModel(BasePerformanceModel):
    def process_trace(self, trace_data):
        # Your implementation
        return PerformanceMetrics(...)
```

### Add to Framework
```python
# In performance_model.py
self.my_model = MyModel(self.config)
self.models.append(self.my_model)
```

## Troubleshooting

### Empty trace data
- Check trace file format
- Verify file path
- Check trace processor logs

### Unrealistic predictions
- Verify microarchitectural parameters
- Check model assumptions
- Calibrate against real hardware

### Import errors
- Install dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify file structure

## File Structure

```
annieflower/
├── analytical/          # Performance models
│   ├── base_model.py
│   ├── bp_model.py
│   ├── cache_model.py
│   ├── rob_model.py
│   ├── lsq_model.py
│   └── memory_model.py
├── traces/              # Instruction traces
├── trace_processor.py    # Trace parser
├── performance_model.py  # Main framework
└── example_usage.py      # Examples
```

## Next Steps

1. **Run walkthrough**: `python walkthrough.py`
2. **Read tutorial**: See `TUTORIAL.md`
3. **Collect traces**: Use Tacit or simulators
4. **Calibrate**: Adjust models to match hardware
5. **Extend**: Add new models or ML correction


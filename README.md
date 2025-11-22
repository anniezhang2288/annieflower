# AnnieFlower: Performance Modeling Framework

A hybrid analytical-ML performance modeling framework inspired by Concorde, using Tacit for instruction trace collection.

## Overview

This framework combines analytical performance models with machine learning to predict CPU performance from instruction traces. Similar to Concorde's approach, it uses compositional modeling where each microarchitectural component (branch predictor, caches, ROB, LSQ, memory) is modeled separately and then combined to predict overall performance.

### Key Features

- **Analytical Models**: Component-wise performance models for:
  - Branch Predictor (BP)
  - Cache Hierarchy (L1/L2/L3)
  - Reorder Buffer (ROB)
  - Load-Store Queue (LSQ)
  - Memory System

- **Tacit Integration**: Processes instruction traces from Tacit
- **Compositional Modeling**: Combines individual component models
- **Configuration Support**: Easy microarchitectural parameter tuning
- **Bottleneck Analysis**: Identifies performance bottlenecks
- **Design Space Exploration**: Compare different configurations

## Repository Structure

```
annieflower/
├── analytical/              # Analytical performance models
│   ├── __init__.py
│   ├── base_model.py        # Base class for all models
│   ├── bp_model.py          # Branch predictor model
│   ├── cache_model.py       # Cache hierarchy model
│   ├── rob_model.py         # Reorder buffer model
│   ├── lsq_model.py         # Load-store queue model
│   └── memory_model.py      # Memory system model
├── traces/                  # Tacit instruction traces
├── trace_processor.py       # Tacit trace parser/processor
├── performance_model.py     # Main orchestration framework
├── example_usage.py         # Usage examples
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- NumPy
- (Optional) SciPy for advanced analysis

### Installation

1. Clone or navigate to the repository:
```bash
cd annieflower
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have Tacit trace files in the `traces/` directory, or modify paths in your code.

## Usage

### Basic Usage

```python
# Import from src package
from src.analytical import MicroarchConfig
from src.performance_model import PerformanceModelFramework

# Create microarchitectural configuration
config = MicroarchConfig(
    pipeline_width=4,
    pipeline_depth=14,
    rob_size=192,
    lsq_size=72,
    l1d_size=32 * 1024,  # 32KB
    l2_size=256 * 1024,   # 256KB
    l3_size=8 * 1024 * 1024  # 8MB
)

# Initialize framework
framework = PerformanceModelFramework(config)

# Process Tacit trace file
results = framework.predict_performance("traces/example_trace.txt")

# Access results
print(f"IPC: {results['performance_breakdown']['ipc']:.3f}")
print(f"Total Cycles: {results['performance_breakdown']['total_cycles']:.0f}")
```

### Custom Configuration

```python
# High-performance configuration
config = MicroarchConfig(
    pipeline_width=6,
    rob_size=256,
    l1d_size=64 * 1024,
    l2_size=512 * 1024,
    l3_size=16 * 1024 * 1024,
    bp_size=8192
)

framework = PerformanceModelFramework(config)
results = framework.process_trace_file("traces/example_trace.txt")
```

### Compare Configurations

```python
baseline = MicroarchConfig(pipeline_width=4, rob_size=192)
large_cache = MicroarchConfig(pipeline_width=4, rob_size=192, l3_size=16*1024*1024)
wide_pipeline = MicroarchConfig(pipeline_width=6, rob_size=256)

framework = PerformanceModelFramework()
comparison = framework.compare_configurations(
    "traces/example_trace.txt",
    [baseline, large_cache, wide_pipeline]
)
```

### Bottleneck Analysis

```python
from src.performance_model import PerformanceModelFramework

framework = PerformanceModelFramework()
analysis = framework.get_bottleneck_analysis("traces/example_trace.txt")

print(f"Primary Bottleneck: {analysis['primary_bottleneck']['component']}")
for rec in analysis['recommendations']:
    print(f"  - {rec}")
```

## Tacit Trace Format

The framework expects Tacit traces in one of these formats:

### JSON Format
```json
[
  {
    "pc": 4194304,
    "instruction_type": "load",
    "memory_address": 8388608,
    "cycle": 0
  },
  {
    "pc": 4194308,
    "instruction_type": "alu",
    "cycle": 1
  }
]
```

### CSV Format
```csv
pc,instruction_type,memory_address,cycle
0x400000,load,0x800000,0
0x400004,alu,,1
```

### Text Format
```
0x400000 load 0x800000
0x400004 alu
0x400008 branch 1
```

The trace processor automatically detects and parses these formats.

## Architecture

### Component Models

Each analytical model follows the same interface:

1. **process_trace()**: Processes trace data and returns `PerformanceMetrics`
2. **estimate_latency()**: Estimates latency contribution
3. **estimate_throughput()**: Estimates throughput impact
4. **get_stall_cycles()**: Calculates stall cycles
5. **get_utilization()**: Calculates resource utilization

### Performance Metrics

Each model returns:
- `latency`: Total latency in cycles
- `throughput`: Throughput impact (0.0 to 1.0)
- `utilization`: Resource utilization (0.0 to 1.0)
- `stall_cycles`: Stall cycles due to this component
- `additional_metrics`: Component-specific metrics

### Combining Models

The framework combines models by:
1. Processing each component model independently
2. Summing stall cycles from all components
3. Taking the minimum throughput (bottleneck)
4. Computing overall IPC = instructions / (instructions + stalls)

## Microarchitectural Parameters

The `MicroarchConfig` class supports:

- **Pipeline**: `pipeline_width`, `pipeline_depth`
- **Branch Predictor**: `bp_size`, `bp_assoc`
- **Caches**: `l1d_size`, `l1d_assoc`, `l1d_latency`, etc.
- **ROB**: `rob_size`
- **LSQ**: `lsq_size`
- **Memory**: `memory_latency`
- **Execution Units**: `alu_count`, `fpu_count`, `load_store_units`

## Extending the Framework

### Adding a New Model

1. Create a new model class inheriting from `BasePerformanceModel`:

```python
from src.analytical.base_model import BasePerformanceModel, PerformanceMetrics

class MyModel(BasePerformanceModel):
    def process_trace(self, trace_data):
        # Your implementation
        return PerformanceMetrics(...)
    
    def estimate_latency(self, trace_data):
        # Your implementation
        return latency
    
    def estimate_throughput(self, trace_data):
        # Your implementation
        return throughput
```

2. Add it to `performance_model.py`:

```python
self.my_model = MyModel(self.config)
self.models.append(self.my_model)
```

### Customizing Trace Processing

Modify `trace_processor.py` to handle different Tacit output formats or add new trace sources.

## Differences from Concorde

While inspired by Concorde, this framework:

1. **Uses Tacit** instead of Concorde's trace collection method
2. **Focuses on analytical models** (ML integration can be added)
3. **Simplified models** for educational/research purposes
4. **Python implementation** (Concorde may use other languages)

## Future Work

- [ ] Machine learning integration for residual error correction
- [ ] Support for more microarchitectural components
- [ ] Parallel trace processing
- [ ] Visualization tools
- [ ] Integration with simulators
- [ ] Calibration against real hardware

## References

- Concorde: Hybrid Analytical-ML Performance Modeling
- Tacit: Instruction Trace Collection Tool

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

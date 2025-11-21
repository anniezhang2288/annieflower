# Setup Guide: Building a Concorde-like Performance Modeling Framework

This guide explains how to set up and use the performance modeling framework, focusing on the setup for performance models similar to Concorde but using Tacit for instruction traces.

## Understanding the Architecture

### Concorde's Approach

Concorde uses a **compositional modeling** approach:
1. **Analytical Models**: Simple, fast models for each microarchitectural component
2. **Performance Distributions**: Compact representations of performance impact
3. **ML Correction**: Machine learning to correct analytical model errors
4. **Rapid Exploration**: Fast design-space exploration

### Our Adaptation

We adapt this approach by:
1. Using **Tacit** for instruction trace collection (instead of Concorde's method)
2. Building **analytical models** for each component
3. **Composing** models to predict overall performance
4. (Future) Adding ML for residual error correction

## Repository Setup

### 1. Directory Structure

The repository is organized as follows:

```
annieflower/
├── analytical/              # Core analytical models
│   ├── base_model.py        # Abstract base class
│   ├── bp_model.py          # Branch predictor
│   ├── cache_model.py       # Cache hierarchy
│   ├── rob_model.py         # Reorder buffer
│   ├── lsq_model.py         # Load-store queue
│   └── memory_model.py      # Memory system
├── traces/                  # Tacit instruction traces
├── trace_processor.py       # Tacit trace parser
├── performance_model.py     # Main framework
└── example_usage.py         # Examples
```

### 2. Component Models

Each model in `analytical/` follows this pattern:

```python
class ComponentModel(BasePerformanceModel):
    def __init__(self, config: MicroarchConfig):
        super().__init__(config)
        # Component-specific initialization
    
    def process_trace(self, trace_data):
        # 1. Extract relevant instructions
        # 2. Compute metrics
        # 3. Return PerformanceMetrics
        pass
    
    def estimate_latency(self, trace_data):
        # Estimate latency contribution
        pass
    
    def estimate_throughput(self, trace_data):
        # Estimate throughput impact
        pass
```

## Setting Up Performance Models

### Step 1: Define Microarchitectural Configuration

Create a `MicroarchConfig` object with your CPU parameters:

```python
from analytical import MicroarchConfig

config = MicroarchConfig(
    # Pipeline
    pipeline_width=4,      # Instructions per cycle
    pipeline_depth=14,     # Pipeline stages
    
    # Branch Predictor
    bp_size=4096,          # Predictor entries
    bp_assoc=4,            # Associativity
    
    # Cache Hierarchy
    l1d_size=32 * 1024,    # 32KB L1 data cache
    l1d_assoc=8,
    l1d_latency=3,         # Cycles
    
    l2_size=256 * 1024,    # 256KB L2
    l2_assoc=8,
    l2_latency=12,
    
    l3_size=8 * 1024 * 1024,  # 8MB L3
    l3_assoc=16,
    l3_latency=40,
    
    # Memory
    memory_latency=100,    # Cycles to main memory
    
    # ROB and LSQ
    rob_size=192,
    lsq_size=72,
    
    # Execution Units
    alu_count=4,
    fpu_count=2,
    load_store_units=2
)
```

### Step 2: Initialize Models

Each model is initialized with the configuration:

```python
from analytical import (
    BranchPredictorModel,
    CacheModel,
    ReorderBufferModel,
    LoadStoreQueueModel,
    MemoryModel
)

bp_model = BranchPredictorModel(config)
cache_model = CacheModel(config)
rob_model = ReorderBufferModel(config)
lsq_model = LoadStoreQueueModel(config)
memory_model = MemoryModel(config)
```

### Step 3: Process Tacit Traces

Load and process Tacit instruction traces:

```python
from trace_processor import TacitTraceProcessor

processor = TacitTraceProcessor()
trace_data = processor.load_trace("traces/my_trace.txt")

# Process with each model
bp_metrics = bp_model.process_trace(trace_data)
cache_metrics = cache_model.process_trace(trace_data)
# ... etc
```

### Step 4: Combine Results

The `PerformanceModelFramework` combines all models:

```python
from performance_model import PerformanceModelFramework

framework = PerformanceModelFramework(config)
results = framework.process_trace_file("traces/my_trace.txt")

# Results include:
# - Overall IPC
# - Stall cycles per component
# - Throughput impact
# - Utilization metrics
```

## How Each Model Works

### Branch Predictor Model (`bp_model.py`)

**What it models:**
- Branch prediction accuracy
- Misprediction penalty (pipeline flush)
- Branch target buffer (BTB) hits/misses

**Key calculations:**
- Prediction accuracy based on predictor size and branch patterns
- Stall cycles = mispredictions × pipeline_depth
- Throughput impact from pipeline flushes

**Analytical approach:**
- Uses predictor size to estimate base accuracy
- Adjusts for branch bias (highly biased branches are easier to predict)
- Models BTB miss rate based on unique branch targets

### Cache Model (`cache_model.py`)

**What it models:**
- L1, L2, L3 hit/miss rates
- Cache access latency
- Memory access latency

**Key calculations:**
- Hit rates using analytical model (working set size vs cache size)
- Total latency = Σ(hits × cache_latency + misses × next_level_latency)
- Stall cycles from cache misses (accounting for OoO execution)

**Analytical approach:**
- Estimates hit rate based on working set size and cache capacity
- Models associativity impact on hit rate
- Accounts for temporal locality

### Reorder Buffer Model (`rob_model.py`)

**What it models:**
- ROB capacity constraints
- Instruction retirement rate
- ROB full stalls

**Key calculations:**
- ROB occupancy = issue_rate × avg_instruction_latency
- Stalls when occupancy > ROB size
- Retirement latency based on retirement width

**Analytical approach:**
- Uses Little's Law: occupancy = arrival_rate × service_time
- Estimates average instruction latency by type
- Models dependency chain impact

### Load-Store Queue Model (`lsq_model.py`)

**What it models:**
- LSQ capacity constraints
- Memory dependency stalls
- Store-to-load forwarding

**Key calculations:**
- LSQ occupancy based on memory operation rate
- Dependency stalls from address conflicts
- Capacity stalls when LSQ is full

**Analytical approach:**
- Models LSQ occupancy similar to ROB
- Detects address conflicts (same cache line)
- Estimates forwarding opportunities

### Memory Model (`memory_model.py`)

**What it models:**
- Memory bandwidth utilization
- Memory contention
- Memory latency

**Key calculations:**
- Bandwidth usage = data_transferred / time
- Contention stalls when bandwidth saturated
- Memory access latency

**Analytical approach:**
- Calculates bandwidth from cache line transfers
- Models contention when bandwidth > 80% utilized
- Accounts for memory controller overhead

## Trace Processing

### Tacit Trace Format

The framework supports multiple formats. The trace processor expects:

**Required fields:**
- `pc`: Program counter (instruction address)
- `instruction_type`: Type of instruction (load, store, branch, alu, etc.)

**Optional fields:**
- `memory_address`: For load/store instructions
- `branch_taken`: For branch instructions
- `dependencies`: List of dependent instruction indices
- `cycle`: Cycle number

### Customizing Trace Processing

To handle different Tacit output formats, modify `trace_processor.py`:

```python
def _load_text_trace(self, trace_file: str):
    # Customize parsing for your Tacit format
    with open(trace_file, 'r') as f:
        for line in f:
            # Parse your specific format
            entry = parse_your_format(line)
            trace_entries.append(entry)
    return trace_entries
```

## Running the Framework

### Basic Workflow

1. **Collect traces with Tacit**:
   ```bash
   # Use Tacit to generate instruction traces
   # Save to traces/ directory
   ```

2. **Configure microarchitecture**:
   ```python
   config = MicroarchConfig(...)  # Your CPU parameters
   ```

3. **Process traces**:
   ```python
   framework = PerformanceModelFramework(config)
   results = framework.process_trace_file("traces/my_trace.txt")
   ```

4. **Analyze results**:
   ```python
   print(f"IPC: {results['performance_breakdown']['ipc']}")
   print(f"Bottleneck: {results['model_results']['CacheModel']['stall_cycles']}")
   ```

### Design Space Exploration

Compare different configurations:

```python
configs = [
    MicroarchConfig(pipeline_width=4, rob_size=192),
    MicroarchConfig(pipeline_width=6, rob_size=256),
    MicroarchConfig(pipeline_width=4, l3_size=16*1024*1024),
]

framework = PerformanceModelFramework()
comparison = framework.compare_configurations("traces/trace.txt", configs)
```

## Extending the Framework

### Adding a New Component Model

1. **Create model file** (`analytical/my_model.py`):
   ```python
   from .base_model import BasePerformanceModel, PerformanceMetrics
   
   class MyModel(BasePerformanceModel):
       def process_trace(self, trace_data):
           # Your implementation
           return PerformanceMetrics(...)
   ```

2. **Add to framework** (`performance_model.py`):
   ```python
   from analytical import MyModel
   
   class PerformanceModelFramework:
       def __init__(self, config):
           # ...
           self.my_model = MyModel(self.config)
           self.models.append(self.my_model)
   ```

3. **Update configuration** (`base_model.py`):
   ```python
   @dataclass
   class MicroarchConfig:
       # ... existing fields
       my_component_size: int = 1024
   ```

### Integrating Machine Learning

To add ML correction (like Concorde):

1. **Collect training data**:
   - Run analytical models on traces
   - Compare with actual measurements
   - Collect residuals (errors)

2. **Train ML model**:
   ```python
   # Extract features from analytical models
   features = extract_features(analytical_results)
   # Train on residuals
   ml_model = train_model(features, residuals)
   ```

3. **Apply correction**:
   ```python
   analytical_prediction = framework.process_trace(trace)
   ml_correction = ml_model.predict(features)
   final_prediction = analytical_prediction + ml_correction
   ```

## Best Practices

1. **Start simple**: Begin with basic configurations and simple traces
2. **Validate models**: Compare predictions with known benchmarks
3. **Iterate**: Refine analytical models based on validation results
4. **Document**: Keep track of model assumptions and parameters
5. **Calibrate**: Adjust model parameters to match your target architecture

## Troubleshooting

### Empty trace data
- Check trace file format matches expected format
- Verify Tacit is generating traces correctly
- Check trace file path

### Unrealistic predictions
- Verify microarchitectural parameters are reasonable
- Check model assumptions match your architecture
- Consider calibrating model parameters

### Performance issues
- Use smaller trace samples for development
- Profile which model is slowest
- Consider caching intermediate results

## Next Steps

1. **Collect Tacit traces** for your target workloads
2. **Calibrate models** against real hardware measurements
3. **Add ML correction** for improved accuracy
4. **Extend models** for additional components
5. **Visualize results** for better insights

For more examples, see `example_usage.py`.


# Complete Tutorial: Performance Modeling with Tacit Traces

This tutorial walks you through every step of using the performance modeling framework, from collecting instruction traces to using the models and understanding how they work.

## Table of Contents

1. [Getting Instruction Traces with Tacit](#1-getting-instruction-traces-with-tacit)
2. [Understanding How the Performance Models Work](#2-understanding-how-the-performance-models-work)
3. [Using the Performance Models](#3-using-the-performance-models)
4. [Next Steps and Advanced Usage](#4-next-steps-and-advanced-usage)

---

## 1. Getting Instruction Traces with Tacit

### What is Tacit?

Tacit is a tool for collecting dynamic instruction traces. It captures the actual instructions executed by a program, including:
- Instruction addresses (program counters)
- Instruction types (load, store, branch, ALU, etc.)
- Memory addresses accessed
- Branch outcomes
- Dependencies between instructions

### Step 1.1: Installing Tacit

**Option A: If Tacit is available as a package**
```bash
# Check if Tacit is available in your package manager
# or download from the Tacit repository
```

**Option B: Using a simulator or tool that generates traces**
```bash
# Many simulators can generate instruction traces
# Examples: gem5, Sniper, Pin, DynamoRIO
```

**Option C: Manual trace generation**
```bash
# You can also manually create traces for testing
# See the trace format section below
```

### Step 1.2: Generating Instruction Traces

#### Method 1: Using Tacit Directly

```bash
# Basic usage (syntax may vary based on Tacit version)
tacit --program ./my_program --output trace.txt

# With options
tacit --program ./my_program \
      --output trace.json \
      --format json \
      --include-memory-addresses \
      --include-dependencies
```

#### Method 2: Using a Simulator (e.g., gem5)

```bash
# Run gem5 with trace generation
./build/ARM/gem5.opt \
    --debug-flags=Exec \
    --debug-file=trace.out \
    configs/example/se.py \
    -c ./my_program
```

#### Method 3: Using Pin (Intel Pin Tool)

```bash
# Use Pin to generate instruction traces
pin -t trace_tool.so -- ./my_program
```

### Step 1.3: Understanding Trace Format

The framework supports three trace formats. Here's what each looks like:

#### JSON Format (Recommended)

```json
[
  {
    "pc": 4194304,
    "instruction_type": "load",
    "memory_address": 8388608,
    "cycle": 0,
    "dependencies": []
  },
  {
    "pc": 4194308,
    "instruction_type": "alu",
    "cycle": 1,
    "dependencies": [0]
  },
  {
    "pc": 4194312,
    "instruction_type": "store",
    "memory_address": 8388612,
    "cycle": 2,
    "dependencies": [1]
  },
  {
    "pc": 4194316,
    "instruction_type": "branch",
    "branch_taken": true,
    "cycle": 3,
    "dependencies": [2]
  }
]
```

#### CSV Format

```csv
pc,instruction_type,memory_address,cycle,dependencies
0x400000,load,0x800000,0,
0x400004,alu,,1,0
0x400008,store,0x800004,2,1
0x40000c,branch,,3,2
```

#### Text Format

```
0x400000 load 0x800000
0x400004 alu
0x400008 store 0x800004
0x40000c branch 1
```

### Step 1.4: Creating a Test Trace

For testing, you can create a simple trace manually:

```python
# create_test_trace.py
import json

trace = [
    {
        "pc": 0x1000,
        "instruction_type": "load",
        "memory_address": 0x2000,
        "cycle": 0
    },
    {
        "pc": 0x1004,
        "instruction_type": "alu",
        "cycle": 1,
        "dependencies": [0]
    },
    {
        "pc": 0x1008,
        "instruction_type": "store",
        "memory_address": 0x2004,
        "cycle": 2,
        "dependencies": [1]
    },
    {
        "pc": 0x100c,
        "instruction_type": "branch",
        "branch_taken": True,
        "cycle": 3
    }
]

with open("traces/test_trace.json", "w") as f:
    json.dump(trace, f, indent=2)

print("Created test trace: traces/test_trace.json")
```

Run it:
```bash
python create_test_trace.py
```

### Step 1.5: Validating Your Trace

```python
# validate_trace.py
from trace_processor import TacitTraceProcessor

processor = TacitTraceProcessor()
try:
    trace_data = processor.load_trace("traces/test_trace.json")
    print(f"✓ Trace loaded successfully: {len(trace_data)} instructions")
    
    # Get statistics
    stats = processor.get_instruction_statistics(trace_data)
    print("\nInstruction Statistics:")
    for inst_type, count in stats.items():
        print(f"  {inst_type}: {count}")
        
except Exception as e:
    print(f"✗ Error loading trace: {e}")
```

---

## 2. Understanding How the Performance Models Work

Let's dive deep into how each model is implemented and the design decisions behind them.

### 2.1: Base Model Architecture

All models inherit from `BasePerformanceModel`. Let's examine the design:

```python
class BasePerformanceModel(ABC):
    def __init__(self, config: MicroarchConfig):
        self.config = config  # Microarchitectural parameters
        self.name = self.__class__.__name__
    
    @abstractmethod
    def process_trace(self, trace_data):
        # Main method: processes trace and returns metrics
        pass
    
    @abstractmethod
    def estimate_latency(self, trace_data):
        # Estimate latency contribution
        pass
    
    @abstractmethod
    def estimate_throughput(self, trace_data):
        # Estimate throughput impact
        pass
```

**Design Decision**: Using an abstract base class ensures:
- Consistent interface across all models
- Easy to add new models
- Type safety and IDE support

### 2.2: Branch Predictor Model - Deep Dive

Let's understand how the branch predictor model works:

#### Step 2.2.1: Extracting Branches

```python
def _extract_branches(self, trace_data):
    branch_types = ['branch', 'jump', 'call', 'ret', 'conditional']
    return [entry for entry in trace_data 
            if entry.get('instruction_type') in branch_types]
```

**Why**: We filter for branch-like instructions because only these affect the branch predictor.

#### Step 2.2.2: Estimating Accuracy

```python
def _estimate_accuracy(self, branches):
    # Base accuracy based on predictor size
    base_accuracy = 0.95
    
    # Adjust for predictor size
    size_factor = min(1.0, np.log2(self.config.bp_size) / 12.0)
    accuracy = base_accuracy * (0.8 + 0.2 * size_factor)
    
    # Adjust for branch bias
    taken_rate = sum(1 for b in branches if b.get('branch_taken', False)) / len(branches)
    if taken_rate > 0.9 or taken_rate < 0.1:
        accuracy += 0.03  # Highly biased = easier to predict
    elif 0.4 < taken_rate < 0.6:
        accuracy -= 0.02  # Unbiased = harder to predict
    
    return min(0.99, max(0.85, accuracy))
```

**Design Decisions**:
1. **Size factor**: Larger predictors generally have higher accuracy (logarithmic relationship)
2. **Bias adjustment**: Highly biased branches (almost always taken/not taken) are easier to predict
3. **Bounds**: Accuracy is bounded between 85% and 99% (realistic range)

**How to improve**: You could:
- Use actual branch history patterns
- Model specific predictor types (gshare, perceptron, etc.)
- Account for branch target buffer (BTB) misses

#### Step 2.2.3: Calculating Misprediction Stalls

```python
def _calculate_misprediction_stalls(self, branches, misprediction_rate):
    mispredictions = len(branches) * misprediction_rate
    return mispredictions * self.misprediction_penalty
```

**Design Decision**: Each misprediction flushes the pipeline, costing `pipeline_depth` cycles.

**Why**: When a branch is mispredicted, all instructions in the pipeline after the branch must be discarded.

### 2.3: Cache Model - Deep Dive

The cache model is more complex because it models a hierarchy.

#### Step 2.3.1: Cache Hit Rate Estimation

```python
def _estimate_cache_hit_rate(self, addresses, cache_size, associativity):
    unique_addresses = len(set(addresses))
    cache_blocks = cache_size // 64  # 64-byte cache lines
    
    # If working set fits in cache, expect high hit rate
    if unique_addresses <= cache_blocks:
        return 0.95
    
    # Calculate reuse distance
    reuse_factor = unique_addresses / cache_blocks
    
    # Higher associativity improves hit rate
    assoc_factor = min(1.0, associativity / 8.0)
    
    # Base hit rate decreases with larger working sets
    base_hit_rate = 0.85 - (min(0.3, np.log2(reuse_factor) * 0.05))
    hit_rate = base_hit_rate + (assoc_factor * 0.1)
    
    return max(0.5, min(0.98, hit_rate))
```

**Design Decisions**:
1. **Working set analysis**: If all unique addresses fit in cache, hit rate is high
2. **Reuse distance**: Larger working sets relative to cache size reduce hit rate
3. **Associativity**: Higher associativity reduces conflict misses
4. **Logarithmic relationship**: Hit rate degrades logarithmically with working set size

**How to improve**: You could:
- Use stack distance analysis
- Model temporal/spatial locality explicitly
- Account for prefetching
- Use actual cache simulation for calibration

#### Step 2.3.2: Cache Hierarchy Modeling

```python
def _calculate_l2_stats(self, memory_accesses, l1_misses):
    # Only L1 misses access L2
    addresses = [entry.get('memory_address', 0) 
                for entry in memory_accesses[-l1_misses:]]
    
    hit_rate = self._estimate_cache_hit_rate(
        addresses, self.l2_config['size'], self.l2_config['assoc']
    )
    
    hits = int(l1_misses * hit_rate)
    misses = l1_misses - hits
    
    return hits, misses
```

**Design Decision**: Only misses from one level access the next level. This is the inclusion property.

**Why**: This models the cache hierarchy correctly - L2 only sees L1 misses, L3 only sees L2 misses.

### 2.4: ROB Model - Deep Dive

The ROB model uses Little's Law to estimate occupancy.

#### Step 2.4.1: Estimating ROB Occupancy

```python
def _estimate_rob_occupancy(self, trace_data):
    # Estimate average instruction latency
    avg_latency = self._estimate_avg_instruction_latency(trace_data)
    
    # Issue rate (limited by pipeline width)
    issue_rate = min(self.config.pipeline_width, 
                    len(trace_data) / max(1, len(trace_data)))
    
    # ROB occupancy = issue_rate * avg_latency (Little's Law)
    occupancy = issue_rate * avg_latency
    
    # Account for dependency chains
    dependency_factor = self._estimate_dependency_impact(trace_data)
    occupancy *= dependency_factor
    
    return min(self.rob_size, occupancy)
```

**Design Decision**: Using **Little's Law**: `Occupancy = Arrival Rate × Service Time`

**Why**: This is a fundamental queuing theory result that accurately models buffer occupancy.

**Components**:
- **Arrival rate**: How fast instructions enter ROB (limited by pipeline width)
- **Service time**: How long instructions stay in ROB (average completion latency)
- **Dependency factor**: Dependency chains increase occupancy

#### Step 2.4.2: ROB Stalls

```python
def _calculate_rob_stalls(self, trace_data, rob_occupancy):
    if rob_occupancy < self.rob_size:
        return 0.0
    
    # Estimate cycles where ROB is full
    overflow = rob_occupancy - self.rob_size
    stall_probability = overflow / self.rob_size
    
    total_cycles = len(trace_data)
    stall_cycles = total_cycles * stall_probability
    
    return stall_cycles
```

**Design Decision**: Stalls occur when ROB occupancy exceeds capacity.

**Why**: When ROB is full, new instructions cannot be issued, causing pipeline stalls.

### 2.5: LSQ Model - Deep Dive

The LSQ model handles memory dependencies and capacity.

#### Step 2.5.1: Memory Dependency Detection

```python
def _calculate_dependency_stalls(self, memory_ops):
    dependency_count = 0
    
    for i, op in enumerate(memory_ops):
        addr = op.get('memory_address')
        op_type = op.get('instruction_type')
        
        # Check for dependencies with previous operations
        for j in range(max(0, i - 10), i):
            prev_op = memory_ops[j]
            prev_addr = prev_op.get('memory_address')
            prev_type = prev_op.get('instruction_type')
            
            # Load after store (potential forwarding)
            if op_type == 'load' and prev_type == 'store':
                if self._addresses_match(addr, prev_addr):
                    dependency_count += 1
```

**Design Decision**: Only check recent operations (last 10) for dependencies.

**Why**: 
- Reduces computational complexity
- Most dependencies are between nearby instructions
- Can be adjusted based on OoO window size

#### Step 2.5.2: Address Matching

```python
def _addresses_match(self, addr1, addr2, cache_line_size=64):
    line1 = addr1 // cache_line_size
    line2 = addr2 // cache_line_size
    return line1 == line2
```

**Design Decision**: Match addresses at cache line granularity.

**Why**: 
- Memory disambiguation happens at cache line level
- More conservative (may overestimate dependencies)
- Matches hardware behavior

### 2.6: Memory Model - Deep Dive

The memory model focuses on bandwidth and contention.

#### Step 2.6.1: Bandwidth Calculation

```python
def _calculate_bandwidth_usage(self, memory_accesses):
    # Estimate execution time
    execution_time_cycles = len(memory_accesses) * 10
    execution_time_seconds = execution_time_cycles / 3.0e9  # 3GHz
    
    # Total data transferred
    bytes_transferred = len(memory_accesses) * self.cache_line_size
    gb_transferred = bytes_transferred / (1024 ** 3)
    
    # Bandwidth usage
    bandwidth_usage = gb_transferred / execution_time_seconds
    
    return bandwidth_usage
```

**Design Decision**: Each memory access transfers a full cache line (64 bytes).

**Why**: This matches actual hardware behavior - caches transfer data in cache line units.

#### Step 2.6.2: Contention Modeling

```python
def _calculate_contention_stalls(self, memory_accesses, bandwidth_usage):
    if bandwidth_usage < self.memory_bandwidth * 0.8:
        return 0.0
    
    # Calculate saturation level
    saturation = min(1.0, bandwidth_usage / self.memory_bandwidth)
    
    # Stalls increase with saturation
    stall_probability = (saturation - 0.8) / 0.2
    stall_probability = max(0.0, min(1.0, stall_probability))
    
    avg_stall = 5.0  # cycles per contended access
    
    return len(memory_accesses) * stall_probability * avg_stall
```

**Design Decision**: Contention only occurs above 80% bandwidth utilization.

**Why**: 
- Memory controllers can handle high utilization
- Stalls become significant near saturation
- Linear model between 80-100% utilization

---

## 3. Using the Performance Models

### Step 3.1: Basic Usage

```python
# basic_usage.py
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework

# Step 1: Configure your CPU architecture
config = MicroarchConfig(
    pipeline_width=4,      # 4-wide pipeline
    pipeline_depth=14,      # 14-stage pipeline
    rob_size=192,           # 192-entry ROB
    lsq_size=72,           # 72-entry LSQ
    l1d_size=32 * 1024,    # 32KB L1 data cache
    l2_size=256 * 1024,    # 256KB L2 cache
    l3_size=8 * 1024 * 1024  # 8MB L3 cache
)

# Step 2: Initialize the framework
framework = PerformanceModelFramework(config)

# Step 3: Process a trace file
results = framework.process_trace_file("traces/test_trace.json")

# Step 4: Access results
print("Performance Results:")
print(f"  IPC: {results['performance_breakdown']['ipc']:.3f}")
print(f"  Total Cycles: {results['performance_breakdown']['total_cycles']:.0f}")
print(f"  Stall Cycles: {results['performance_breakdown']['total_stall_cycles']:.0f}")

print("\nComponent Breakdown:")
for component, metrics in results['model_results'].items():
    print(f"\n  {component}:")
    print(f"    Stall Cycles: {metrics['stall_cycles']:.2f}")
    print(f"    Throughput: {metrics['throughput']:.3f}")
    print(f"    Utilization: {metrics['utilization']:.3f}")
```

### Step 3.2: Understanding the Results

The `process_trace_file()` method returns a dictionary with:

```python
{
    'performance_breakdown': {
        'total_instructions': 1000,
        'total_cycles': 1250,
        'total_stall_cycles': 250,
        'ipc': 0.8,  # Instructions Per Cycle
        'overall_throughput': 0.75,
        'stall_breakdown': {
            'BranchPredictorModel': 50,
            'CacheModel': 100,
            'ReorderBufferModel': 30,
            'LoadStoreQueueModel': 40,
            'MemoryModel': 30
        }
    },
    'model_results': {
        'BranchPredictorModel': {
            'latency': 100,
            'throughput': 0.95,
            'utilization': 0.15,
            'stall_cycles': 50,
            'additional_metrics': {
                'prediction_accuracy': 0.96,
                'misprediction_rate': 0.04,
                'branch_count': 200
            }
        },
        # ... other models
    },
    'trace_statistics': {
        'load': 300,
        'store': 200,
        'alu': 400,
        'branch': 100
    }
}
```

### Step 3.3: Customizing Microarchitectural Parameters

```python
# custom_config.py
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework

# High-performance configuration
high_perf_config = MicroarchConfig(
    pipeline_width=6,           # Wider pipeline
    pipeline_depth=16,           # Deeper pipeline
    rob_size=256,               # Larger ROB
    lsq_size=96,                # Larger LSQ
    l1d_size=64 * 1024,         # 64KB L1D (2x)
    l1d_assoc=16,               # Higher associativity
    l2_size=512 * 1024,         # 512KB L2 (2x)
    l3_size=16 * 1024 * 1024,   # 16MB L3 (2x)
    bp_size=8192,               # Larger branch predictor
    memory_latency=80           # Lower memory latency
)

framework = PerformanceModelFramework(high_perf_config)
results = framework.process_trace_file("traces/test_trace.json")
print(f"High-perf IPC: {results['performance_breakdown']['ipc']:.3f}")
```

### Step 3.4: Comparing Configurations

```python
# compare_configs.py
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework

# Baseline
baseline = MicroarchConfig(
    pipeline_width=4,
    rob_size=192,
    l1d_size=32 * 1024,
    l2_size=256 * 1024,
    l3_size=8 * 1024 * 1024
)

# Larger caches
large_cache = MicroarchConfig(
    pipeline_width=4,
    rob_size=192,
    l1d_size=64 * 1024,      # 2x L1D
    l2_size=512 * 1024,       # 2x L2
    l3_size=16 * 1024 * 1024  # 2x L3
)

# Wider pipeline
wide_pipeline = MicroarchConfig(
    pipeline_width=6,         # Wider
    rob_size=256,             # Larger ROB
    l1d_size=32 * 1024,
    l2_size=256 * 1024,
    l3_size=8 * 1024 * 1024
)

framework = PerformanceModelFramework()
comparison = framework.compare_configurations(
    "traces/test_trace.json",
    [baseline, large_cache, wide_pipeline]
)

print("Configuration Comparison:")
for config_name, config_data in comparison.items():
    ipc = config_data['performance']['ipc']
    config = config_data['config']
    print(f"\n{config_name}:")
    print(f"  IPC: {ipc:.3f}")
    print(f"  Pipeline Width: {config['pipeline_width']}")
    print(f"  L3 Size: {config['l3_size'] / (1024*1024):.0f}MB")
```

### Step 3.5: Bottleneck Analysis

```python
# bottleneck_analysis.py
from performance_model import PerformanceModelFramework

framework = PerformanceModelFramework()
analysis = framework.get_bottleneck_analysis("traces/test_trace.json")

print("Bottleneck Analysis:")
print(f"\nPrimary Bottleneck: {analysis['primary_bottleneck']['component']}")
print(f"  Stall Cycles: {analysis['primary_bottleneck']['stall_cycles']:.2f}")
print(f"  Percentage: {analysis['primary_bottleneck']['percentage']:.1f}%")

print(f"\nThroughput Bottleneck: {analysis['throughput_bottleneck']['component']}")
print(f"  Throughput: {analysis['throughput_bottleneck']['throughput']:.3f}")

print(f"\nUtilization Bottleneck: {analysis['utilization_bottleneck']['component']}")
print(f"  Utilization: {analysis['utilization_bottleneck']['utilization']:.3f}")

print("\nRecommendations:")
for rec in analysis['recommendations']:
    print(f"  • {rec}")
```

### Step 3.6: Processing Traces Directly (Without Files)

```python
# direct_trace_processing.py
from performance_model import PerformanceModelFramework

# Create trace data directly (e.g., from Tacit API)
trace_data = [
    {'pc': 0x1000, 'instruction_type': 'load', 'memory_address': 0x2000},
    {'pc': 0x1004, 'instruction_type': 'alu'},
    {'pc': 0x1008, 'instruction_type': 'store', 'memory_address': 0x2004},
    {'pc': 0x100c, 'instruction_type': 'branch', 'branch_taken': True},
]

framework = PerformanceModelFramework()
results = framework.process_trace(trace_data)

print(f"IPC: {results['performance_breakdown']['ipc']:.3f}")
```

### Step 3.7: Saving and Loading Results

```python
# save_results.py
import json
from performance_model import PerformanceModelFramework

framework = PerformanceModelFramework()

# Process and save
results = framework.predict_performance(
    "traces/test_trace.json",
    output_file="results.json"
)

# Later, load results
with open("results.json", "r") as f:
    loaded_results = json.load(f)

print(f"Loaded IPC: {loaded_results['performance_breakdown']['ipc']:.3f}")
```

---

## 4. Next Steps and Advanced Usage

### Step 4.1: Calibrating Models

To improve accuracy, calibrate models against real hardware:

```python
# calibrate_models.py
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework

# Known benchmark results
benchmark_results = {
    "spec2006_bzip2": {"actual_ipc": 1.2},
    "spec2006_mcf": {"actual_ipc": 0.8},
    # ... more benchmarks
}

# Your predictions
config = MicroarchConfig()
framework = PerformanceModelFramework(config)

calibration_data = []
for benchmark, actual in benchmark_results.items():
    trace_file = f"traces/{benchmark}.json"
    predicted = framework.process_trace_file(trace_file)
    predicted_ipc = predicted['performance_breakdown']['ipc']
    
    error = abs(predicted_ipc - actual['actual_ipc'])
    calibration_data.append({
        'benchmark': benchmark,
        'predicted': predicted_ipc,
        'actual': actual['actual_ipc'],
        'error': error
    })

# Analyze errors and adjust model parameters
for data in calibration_data:
    print(f"{data['benchmark']}: Predicted={data['predicted']:.3f}, "
          f"Actual={data['actual']:.3f}, Error={data['error']:.3f}")
```

### Step 4.2: Adding Machine Learning Correction

Following Concorde's approach, add ML to correct analytical model errors:

```python
# ml_correction.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from performance_model import PerformanceModelFramework

# Step 1: Collect training data
def extract_features(analytical_results):
    """Extract features from analytical model results."""
    features = []
    for model_name, metrics in analytical_results['model_results'].items():
        features.extend([
            metrics['stall_cycles'],
            metrics['throughput'],
            metrics['utilization']
        ])
        # Add additional metrics
        for key, value in metrics.get('additional_metrics', {}).items():
            if isinstance(value, (int, float)):
                features.append(value)
    return features

# Step 2: Train ML model
def train_ml_model(training_traces, actual_ipcs):
    """Train ML model to predict residual errors."""
    X = []
    y = []
    
    framework = PerformanceModelFramework()
    
    for trace_file, actual_ipc in zip(training_traces, actual_ipcs):
        results = framework.process_trace_file(trace_file)
        features = extract_features(results)
        predicted_ipc = results['performance_breakdown']['ipc']
        
        X.append(features)
        y.append(actual_ipc - predicted_ipc)  # Residual
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# Step 3: Use ML correction
def predict_with_ml(trace_file, ml_model):
    """Predict performance with ML correction."""
    framework = PerformanceModelFramework()
    analytical_results = framework.process_trace_file(trace_file)
    
    # Get analytical prediction
    analytical_ipc = analytical_results['performance_breakdown']['ipc']
    
    # Get ML correction
    features = extract_features(analytical_results)
    ml_correction = ml_model.predict([features])[0]
    
    # Final prediction
    final_ipc = analytical_ipc + ml_correction
    
    return {
        'analytical_ipc': analytical_ipc,
        'ml_correction': ml_correction,
        'final_ipc': final_ipc
    }
```

### Step 4.3: Extending Models

Add a new component model:

```python
# analytical/execution_unit_model.py
from .base_model import BasePerformanceModel, PerformanceMetrics

class ExecutionUnitModel(BasePerformanceModel):
    """Model for execution unit utilization and stalls."""
    
    def __init__(self, config):
        super().__init__(config)
        self.alu_count = config.alu_count
        self.fpu_count = config.fpu_count
    
    def process_trace(self, trace_data):
        alu_ops = [e for e in trace_data if e.get('instruction_type') == 'alu']
        fpu_ops = [e for e in trace_data if e.get('instruction_type') == 'fpu']
        
        # Calculate utilization
        alu_utilization = len(alu_ops) / (len(trace_data) * self.alu_count)
        fpu_utilization = len(fpu_ops) / (len(trace_data) * self.fpu_count)
        
        # Calculate stalls from resource conflicts
        stalls = self._calculate_resource_stalls(alu_ops, fpu_ops)
        
        return PerformanceMetrics(
            latency=0.0,  # Execution units don't add latency
            throughput=min(1.0, 1.0 - (stalls / len(trace_data))),
            utilization=max(alu_utilization, fpu_utilization),
            stall_cycles=stalls
        )
    
    def _calculate_resource_stalls(self, alu_ops, fpu_ops):
        # Model resource conflicts
        # Simplified: stalls occur when demand > supply
        alu_demand = len(alu_ops)
        fpu_demand = len(fpu_ops)
        
        alu_stalls = max(0, (alu_demand - self.alu_count * len(alu_ops)) * 0.1)
        fpu_stalls = max(0, (fpu_demand - self.fpu_count * len(fpu_ops)) * 0.1)
        
        return alu_stalls + fpu_stalls
    
    def estimate_latency(self, trace_data):
        return 0.0
    
    def estimate_throughput(self, trace_data):
        metrics = self.process_trace(trace_data)
        return metrics.throughput
```

Then add it to `performance_model.py`:

```python
from analytical import ExecutionUnitModel

class PerformanceModelFramework:
    def __init__(self, config=None):
        # ... existing models ...
        self.exec_unit_model = ExecutionUnitModel(self.config)
        self.models.append(self.exec_unit_model)
```

### Step 4.4: Batch Processing

Process multiple traces:

```python
# batch_processing.py
from pathlib import Path
from performance_model import PerformanceModelFramework
import json

def process_trace_directory(trace_dir, output_dir):
    """Process all traces in a directory."""
    framework = PerformanceModelFramework()
    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_summary = []
    
    for trace_file in trace_dir.glob("*.json"):
        print(f"Processing {trace_file.name}...")
        
        results = framework.process_trace_file(str(trace_file))
        
        # Save individual results
        output_file = output_dir / f"{trace_file.stem}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Add to summary
        results_summary.append({
            'trace': trace_file.name,
            'ipc': results['performance_breakdown']['ipc'],
            'total_cycles': results['performance_breakdown']['total_cycles']
        })
    
    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return results_summary

# Usage
summary = process_trace_directory("traces/", "results/")
print(f"Processed {len(summary)} traces")
```

### Step 4.5: Visualization

Create visualizations of results:

```python
# visualize_results.py
import matplotlib.pyplot as plt
from performance_model import PerformanceModelFramework

def plot_stall_breakdown(trace_file):
    """Plot stall cycle breakdown by component."""
    framework = PerformanceModelFramework()
    results = framework.process_trace_file(trace_file)
    
    stall_breakdown = results['performance_breakdown']['stall_breakdown']
    
    components = list(stall_breakdown.keys())
    stalls = list(stall_breakdown.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(components, stalls)
    plt.xlabel('Component')
    plt.ylabel('Stall Cycles')
    plt.title('Stall Cycle Breakdown')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('stall_breakdown.png')
    print("Saved stall_breakdown.png")

def plot_config_comparison(trace_file, configs, config_names):
    """Compare IPC across different configurations."""
    framework = PerformanceModelFramework()
    comparison = framework.compare_configurations(trace_file, configs)
    
    ipcs = [comparison[f'config_{i}']['performance']['ipc'] 
            for i in range(len(configs))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(config_names, ipcs)
    plt.xlabel('Configuration')
    plt.ylabel('IPC')
    plt.title('IPC Comparison Across Configurations')
    plt.tight_layout()
    plt.savefig('config_comparison.png')
    print("Saved config_comparison.png")

# Usage
plot_stall_breakdown("traces/test_trace.json")
```

---

## Summary

You now know:

1. **How to get traces**: Use Tacit, simulators, or create test traces
2. **How models work**: Each model uses analytical techniques (Little's Law, hit rate estimation, etc.)
3. **How to use them**: Process traces, compare configurations, analyze bottlenecks
4. **Next steps**: Calibrate, add ML, extend models, visualize results

The framework is designed to be:
- **Extensible**: Easy to add new models
- **Composable**: Models work independently and combine naturally
- **Fast**: Analytical models are much faster than simulation
- **Interpretable**: You can understand why predictions are made

Start with simple traces, validate against known benchmarks, and iteratively improve the models!


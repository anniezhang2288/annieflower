# Architecture Overview: Concorde-like Performance Modeling with Tacit

## High-Level Architecture

This framework implements a **compositional performance modeling** approach inspired by Concorde, adapted to use Tacit for instruction trace collection.

```
┌─────────────────┐
│  Tacit Traces   │  ← Instruction traces from Tacit
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trace Processor │  ← Parse and normalize traces
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Analytical Performance Models    │
├─────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐        │
│  │    BP    │  │  Cache   │        │
│  │  Model   │  │  Model   │        │
│  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐        │
│  │   ROB    │  │   LSQ    │        │
│  │  Model   │  │  Model   │        │
│  └──────────┘  └──────────┘        │
│  ┌──────────┐                      │
│  │ Memory   │                      │
│  │  Model   │                      │
│  └──────────┘                      │
└────────┬───────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Composition   │  ← Combine model results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Performance     │  ← IPC, stalls, bottlenecks
│ Predictions     │
└─────────────────┘
```

## Comparison with Concorde

### Similarities

1. **Compositional Modeling**: Both decompose the CPU into components
2. **Analytical Models**: Both use fast analytical models for each component
3. **Performance Distributions**: Both represent performance impact compactly
4. **Rapid Exploration**: Both enable fast design-space exploration

### Differences

| Aspect | Concorde | This Framework |
|--------|----------|----------------|
| **Trace Source** | Concorde's method | Tacit |
| **Implementation** | (Unknown) | Python |
| **ML Integration** | Yes | (Future work) |
| **Focus** | Production-ready | Research/Educational |

## Component Model Architecture

Each analytical model follows this structure:

```python
class ComponentModel(BasePerformanceModel):
    def process_trace(trace_data):
        # 1. Extract relevant instructions
        relevant = filter_instructions(trace_data)
        
        # 2. Compute analytical metrics
        hit_rate = estimate_hit_rate(relevant)
        latency = calculate_latency(relevant, hit_rate)
        stalls = calculate_stalls(relevant, hit_rate)
        
        # 3. Return PerformanceMetrics
        return PerformanceMetrics(
            latency=latency,
            throughput=throughput,
            utilization=utilization,
            stall_cycles=stalls
        )
```

### Model Responsibilities

Each model is responsible for:

1. **Filtering**: Extract relevant instructions from trace
2. **Analysis**: Compute component-specific metrics
3. **Prediction**: Estimate latency, throughput, stalls
4. **Metrics**: Return standardized performance metrics

## Data Flow

### 1. Trace Collection (Tacit)

```
Application → Tacit → Instruction Trace File
```

Tacit generates traces containing:
- Instruction addresses (PC)
- Instruction types
- Memory addresses (for loads/stores)
- Branch outcomes
- Dependencies

### 2. Trace Processing

```
Trace File → TraceProcessor → Normalized Trace Data
```

The trace processor:
- Parses various formats (JSON, CSV, text)
- Normalizes to standard format
- Validates trace data

### 3. Model Processing

```
Trace Data → [BP Model] → BP Metrics
Trace Data → [Cache Model] → Cache Metrics
Trace Data → [ROB Model] → ROB Metrics
Trace Data → [LSQ Model] → LSQ Metrics
Trace Data → [Memory Model] → Memory Metrics
```

Each model processes the trace independently.

### 4. Composition

```
BP Metrics ─┐
Cache Metrics ─┤
ROB Metrics ──┼→ Composition → Overall Performance
LSQ Metrics ──┤
Memory Metrics ─┘
```

Composition combines:
- **Stall Cycles**: Sum from all components
- **Throughput**: Minimum (bottleneck)
- **IPC**: Instructions / (Instructions + Stalls)

## Model Details

### Branch Predictor Model

**Input**: Branch instructions from trace

**Analysis**:
- Predictor size → Base accuracy
- Branch patterns → Accuracy adjustment
- Mispredictions → Pipeline flushes

**Output**:
- Prediction accuracy
- Misprediction stalls
- Throughput impact

### Cache Model

**Input**: Memory access instructions

**Analysis**:
- Working set size vs cache size → Hit rate
- Associativity → Hit rate adjustment
- Misses → Next level access

**Output**:
- L1/L2/L3 hit rates
- Cache access latency
- Memory access count

### ROB Model

**Input**: All instructions

**Analysis**:
- Issue rate × Avg latency → Occupancy
- Occupancy vs ROB size → Stalls
- Retirement width → Retirement rate

**Output**:
- ROB occupancy
- Capacity stalls
- Retirement latency

### LSQ Model

**Input**: Load/store instructions

**Analysis**:
- Memory op rate × Latency → Occupancy
- Address conflicts → Dependency stalls
- LSQ capacity → Capacity stalls

**Output**:
- LSQ occupancy
- Dependency stalls
- Capacity stalls

### Memory Model

**Input**: Memory accesses (L3 misses)

**Analysis**:
- Data transferred / Time → Bandwidth
- Bandwidth saturation → Contention
- Memory latency → Access time

**Output**:
- Bandwidth utilization
- Contention stalls
- Memory latency

## Configuration System

The `MicroarchConfig` dataclass encapsulates all microarchitectural parameters:

```python
@dataclass
class MicroarchConfig:
    # Pipeline
    pipeline_width: int = 4
    pipeline_depth: int = 14
    
    # Components
    rob_size: int = 192
    lsq_size: int = 72
    bp_size: int = 4096
    
    # Caches
    l1d_size: int = 32 * 1024
    l2_size: int = 256 * 1024
    l3_size: int = 8 * 1024 * 1024
    
    # Memory
    memory_latency: int = 100
```

This enables:
- Easy configuration changes
- Design space exploration
- Configuration comparison

## Extensibility

### Adding New Models

1. Inherit from `BasePerformanceModel`
2. Implement required methods
3. Add to `PerformanceModelFramework`
4. Update `MicroarchConfig` if needed

### Integrating ML

Future ML integration would:

1. **Collect residuals**: Analytical prediction - Actual measurement
2. **Extract features**: From analytical models and traces
3. **Train model**: Predict residuals from features
4. **Apply correction**: Analytical + ML correction

This follows Concorde's hybrid approach.

## Performance Characteristics

### Analytical Models

- **Fast**: O(n) where n = trace length
- **Interpretable**: Clear cause-effect relationships
- **Composable**: Independent components

### Composition

- **Linear**: O(m) where m = number of models
- **Parallelizable**: Models can run in parallel
- **Scalable**: Easy to add more models

## Validation Strategy

1. **Unit Tests**: Each model independently
2. **Integration Tests**: Full framework
3. **Calibration**: Adjust parameters to match hardware
4. **Benchmarking**: Compare predictions with measurements

## Future Enhancements

1. **ML Correction**: Add ML models for residual error
2. **More Components**: Add models for more components
3. **Parallel Processing**: Process traces in parallel
4. **Visualization**: Visualize predictions and bottlenecks
5. **Simulator Integration**: Connect with cycle-accurate simulators


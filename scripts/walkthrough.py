"""
Interactive Walkthrough: Step-by-step demonstration of the performance modeling framework.
Run this script to see how everything works together.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from src package
from src.analytical import MicroarchConfig
from src.performance_model import PerformanceModelFramework
from src.trace_processor import TacitTraceProcessor


def step_1_create_test_trace():
    """Step 1: Create a test instruction trace."""
    print("=" * 70)
    print("STEP 1: Creating a Test Instruction Trace")
    print("=" * 70)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create traces directory if it doesn't exist
    traces_dir = project_root / "traces"
    traces_dir.mkdir(exist_ok=True)
    
    # Create a realistic trace with various instruction types
    trace = []
    
    # Simulate a loop with memory accesses
    for i in range(100):
        # Load instruction
        trace.append({
            "pc": 0x1000 + i * 4,
            "instruction_type": "load",
            "memory_address": 0x2000 + i * 8,
            "cycle": i * 2,
            "dependencies": [i-1] if i > 0 else []
        })
        
        # ALU instruction (depends on load)
        trace.append({
            "pc": 0x1004 + i * 4,
            "instruction_type": "alu",
            "cycle": i * 2 + 1,
            "dependencies": [i * 2]
        })
        
        # Store instruction (depends on ALU)
        trace.append({
            "pc": 0x1008 + i * 4,
            "instruction_type": "store",
            "memory_address": 0x3000 + i * 8,
            "cycle": i * 2 + 2,
            "dependencies": [i * 2 + 1]
        })
        
        # Branch every 10 iterations
        if i % 10 == 9:
            trace.append({
                "pc": 0x100c + i * 4,
                "instruction_type": "branch",
                "branch_taken": True,
                "cycle": i * 2 + 3,
                "dependencies": [i * 2 + 2]
            })
    
    # Save trace
    trace_file = traces_dir / "walkthrough_trace.json"
    with open(trace_file, "w") as f:
        json.dump(trace, f, indent=2)
    
    print(f"✓ Created test trace: {trace_file}")
    print(f"  Total instructions: {len(trace)}")
    print(f"  Instruction types: load, alu, store, branch")
    print()
    
    return str(trace_file)


def step_2_validate_trace(trace_file):
    """Step 2: Validate and inspect the trace."""
    print("=" * 70)
    print("STEP 2: Validating and Inspecting the Trace")
    print("=" * 70)
    
    processor = TacitTraceProcessor()
    trace_data = processor.load_trace(trace_file)
    
    print(f"✓ Trace loaded successfully")
    print(f"  Total instructions: {len(trace_data)}")
    
    # Get statistics
    stats = processor.get_instruction_statistics(trace_data)
    print("\nInstruction Statistics:")
    for inst_type, count in sorted(stats.items()):
        percentage = (count / len(trace_data)) * 100
        print(f"  {inst_type:12s}: {count:4d} ({percentage:5.1f}%)")
    
    # Show first few instructions
    print("\nFirst 5 instructions:")
    for i, entry in enumerate(trace_data[:5]):
        print(f"  [{i}] PC=0x{entry['pc']:x}, Type={entry['instruction_type']}, "
              f"Cycle={entry.get('cycle', 'N/A')}")
    
    print()
    return trace_data


def step_3_configure_microarchitecture():
    """Step 3: Configure microarchitectural parameters."""
    print("=" * 70)
    print("STEP 3: Configuring Microarchitectural Parameters")
    print("=" * 70)
    
    # Create a realistic configuration (similar to modern CPU)
    config = MicroarchConfig(
        # Pipeline
        pipeline_width=4,      # 4 instructions per cycle
        pipeline_depth=14,      # 14-stage pipeline
        
        # Branch Predictor
        bp_size=4096,          # 4K entries
        bp_assoc=4,            # 4-way associative
        
        # Cache Hierarchy
        l1d_size=32 * 1024,    # 32KB L1 data cache
        l1d_assoc=8,           # 8-way associative
        l1d_latency=3,         # 3 cycles
        
        l1i_size=32 * 1024,    # 32KB L1 instruction cache
        l1i_assoc=8,
        l1i_latency=3,
        
        l2_size=256 * 1024,   # 256KB L2 cache
        l2_assoc=8,
        l2_latency=12,         # 12 cycles
        
        l3_size=8 * 1024 * 1024,  # 8MB L3 cache
        l3_assoc=16,
        l3_latency=40,         # 40 cycles
        
        # Memory
        memory_latency=100,    # 100 cycles to main memory
        
        # ROB and LSQ
        rob_size=192,          # 192-entry reorder buffer
        lsq_size=72,           # 72-entry load-store queue
        
        # Execution Units
        alu_count=4,
        fpu_count=2,
        load_store_units=2
    )
    
    print("✓ Microarchitectural Configuration:")
    print(f"  Pipeline: {config.pipeline_width}-wide, {config.pipeline_depth} stages")
    print(f"  ROB: {config.rob_size} entries")
    print(f"  LSQ: {config.lsq_size} entries")
    print(f"  L1D: {config.l1d_size // 1024}KB, {config.l1d_assoc}-way")
    print(f"  L2:  {config.l2_size // 1024}KB, {config.l2_assoc}-way")
    print(f"  L3:  {config.l3_size // (1024*1024)}MB, {config.l3_assoc}-way")
    print(f"  Memory Latency: {config.memory_latency} cycles")
    print()
    
    return config


def step_4_initialize_models(config):
    """Step 4: Initialize performance models."""
    print("=" * 70)
    print("STEP 4: Initializing Performance Models")
    print("=" * 70)
    
    framework = PerformanceModelFramework(config)
    
    print("✓ Initialized Performance Model Framework")
    print(f"  Models loaded: {len(framework.models)}")
    for model in framework.models:
        print(f"    - {model.name}")
    print()
    
    return framework


def step_5_process_trace(framework, trace_file):
    """Step 5: Process trace and get predictions."""
    print("=" * 70)
    print("STEP 5: Processing Trace and Getting Predictions")
    print("=" * 70)
    
    results = framework.process_trace_file(trace_file)
    
    # Overall performance
    perf = results['performance_breakdown']
    print("✓ Performance Prediction Results:")
    print(f"  Total Instructions: {perf['total_instructions']}")
    print(f"  Total Cycles:       {perf['total_cycles']:.0f}")
    print(f"  Total Stall Cycles: {perf['total_stall_cycles']:.2f}")
    print(f"  IPC:                {perf['ipc']:.3f}")
    print(f"  Overall Throughput: {perf['overall_throughput']:.3f}")
    print()
    
    # Component breakdown
    print("Component Breakdown:")
    for component, metrics in results['model_results'].items():
        print(f"\n  {component}:")
        print(f"    Stall Cycles:  {metrics['stall_cycles']:8.2f} cycles")
        print(f"    Throughput:    {metrics['throughput']:8.3f}")
        print(f"    Utilization:   {metrics['utilization']:8.3f}")
        
        # Show additional metrics
        if metrics['additional_metrics']:
            print(f"    Additional Metrics:")
            for key, value in list(metrics['additional_metrics'].items())[:3]:
                if isinstance(value, float):
                    print(f"      {key}: {value:.3f}")
                else:
                    print(f"      {key}: {value}")
    
    print()
    return results


def step_6_analyze_bottlenecks(framework, trace_file):
    """Step 6: Analyze performance bottlenecks."""
    print("=" * 70)
    print("STEP 6: Analyzing Performance Bottlenecks")
    print("=" * 70)
    
    analysis = framework.get_bottleneck_analysis(trace_file)
    
    print("✓ Bottleneck Analysis:")
    print(f"\n  Primary Bottleneck:")
    print(f"    Component:  {analysis['primary_bottleneck']['component']}")
    print(f"    Stall Cycles: {analysis['primary_bottleneck']['stall_cycles']:.2f}")
    print(f"    Percentage:  {analysis['primary_bottleneck']['percentage']:.1f}%")
    
    print(f"\n  Throughput Bottleneck:")
    print(f"    Component:  {analysis['throughput_bottleneck']['component']}")
    print(f"    Throughput: {analysis['throughput_bottleneck']['throughput']:.3f}")
    
    print(f"\n  Utilization Bottleneck:")
    print(f"    Component:   {analysis['utilization_bottleneck']['component']}")
    print(f"    Utilization: {analysis['utilization_bottleneck']['utilization']:.3f}")
    
    if analysis['recommendations']:
        print(f"\n  Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"    {i}. {rec}")
    
    print()


def step_7_compare_configurations(framework, trace_file):
    """Step 7: Compare different configurations."""
    print("=" * 70)
    print("STEP 7: Comparing Different Configurations")
    print("=" * 70)
    
    # Baseline configuration
    baseline = MicroarchConfig(
        pipeline_width=4,
        rob_size=192,
        l1d_size=32 * 1024,
        l2_size=256 * 1024,
        l3_size=8 * 1024 * 1024
    )
    
    # Configuration with larger caches
    large_cache = MicroarchConfig(
        pipeline_width=4,
        rob_size=192,
        l1d_size=64 * 1024,      # 2x L1D
        l2_size=512 * 1024,      # 2x L2
        l3_size=16 * 1024 * 1024  # 2x L3
    )
    
    # Configuration with wider pipeline
    wide_pipeline = MicroarchConfig(
        pipeline_width=6,       # Wider
        rob_size=256,           # Larger ROB
        l1d_size=32 * 1024,
        l2_size=256 * 1024,
        l3_size=8 * 1024 * 1024
    )
    
    configs = [baseline, large_cache, wide_pipeline]
    config_names = ["Baseline", "Large Cache", "Wide Pipeline"]
    
    comparison = framework.compare_configurations(trace_file, configs)
    
    print("✓ Configuration Comparison:")
    print()
    print(f"{'Configuration':<20} {'IPC':<10} {'Total Cycles':<15} {'Stall Cycles':<15}")
    print("-" * 70)
    
    for i, (name, config_data) in enumerate(zip(config_names, comparison.values())):
        perf = config_data['performance']
        print(f"{name:<20} {perf['ipc']:<10.3f} {perf['total_cycles']:<15.0f} "
              f"{perf['total_stall_cycles']:<15.2f}")
    
    print()
    
    # Find best configuration
    best_config = max(comparison.items(), 
                     key=lambda x: x[1]['performance']['ipc'])
    best_idx = int(best_config[0].split('_')[1])
    print(f"Best Configuration: {config_names[best_idx]} "
          f"(IPC = {best_config[1]['performance']['ipc']:.3f})")
    print()


def step_8_individual_model_demo(framework, trace_file):
    """Step 8: Demonstrate individual model usage."""
    print("=" * 70)
    print("STEP 8: Using Individual Models")
    print("=" * 70)
    
    processor = TacitTraceProcessor()
    trace_data = processor.load_trace(trace_file)
    
    print("✓ Individual Model Demonstrations:")
    print()
    
    # Branch Predictor Model
    print("Branch Predictor Model:")
    bp_metrics = framework.bp_model.process_trace(trace_data)
    print(f"  Prediction Accuracy: {bp_metrics.additional_metrics.get('prediction_accuracy', 0):.3f}")
    print(f"  Misprediction Rate:  {bp_metrics.additional_metrics.get('misprediction_rate', 0):.3f}")
    print(f"  Branch Count:        {bp_metrics.additional_metrics.get('branch_count', 0)}")
    print(f"  Stall Cycles:        {bp_metrics.stall_cycles:.2f}")
    print()
    
    # Cache Model
    print("Cache Model:")
    cache_metrics = framework.cache_model.process_trace(trace_data)
    print(f"  L1 Hit Rate:         {cache_metrics.additional_metrics.get('l1_hit_rate', 0):.3f}")
    print(f"  L2 Hit Rate:         {cache_metrics.additional_metrics.get('l2_hit_rate', 0):.3f}")
    print(f"  L3 Hit Rate:         {cache_metrics.additional_metrics.get('l3_hit_rate', 0):.3f}")
    print(f"  Memory Access Rate:  {cache_metrics.additional_metrics.get('memory_access_rate', 0):.3f}")
    print(f"  Stall Cycles:        {cache_metrics.stall_cycles:.2f}")
    print()
    
    # ROB Model
    print("Reorder Buffer Model:")
    rob_metrics = framework.rob_model.process_trace(trace_data)
    print(f"  Avg ROB Occupancy:   {rob_metrics.additional_metrics.get('avg_rob_occupancy', 0):.1f}")
    print(f"  Max ROB Occupancy:   {rob_metrics.additional_metrics.get('max_rob_occupancy', 0):.1f}")
    print(f"  Utilization:         {rob_metrics.utilization:.3f}")
    print(f"  Stall Cycles:        {rob_metrics.stall_cycles:.2f}")
    print()


def main():
    """Run the complete walkthrough."""
    print("\n" + "=" * 70)
    print("PERFORMANCE MODELING FRAMEWORK - COMPLETE WALKTHROUGH")
    print("=" * 70)
    print("\nThis walkthrough demonstrates:")
    print("  1. Creating instruction traces")
    print("  2. Validating traces")
    print("  3. Configuring microarchitecture")
    print("  4. Initializing models")
    print("  5. Processing traces")
    print("  6. Analyzing bottlenecks")
    print("  7. Comparing configurations")
    print("  8. Using individual models")
    print()
    
    try:
        # Step 1: Create test trace
        trace_file = step_1_create_test_trace()
        
        # Step 2: Validate trace
        trace_data = step_2_validate_trace(trace_file)
        
        # Step 3: Configure microarchitecture
        config = step_3_configure_microarchitecture()
        
        # Step 4: Initialize models
        framework = step_4_initialize_models(config)
        
        # Step 5: Process trace
        results = step_5_process_trace(framework, trace_file)
        
        # Step 6: Analyze bottlenecks
        step_6_analyze_bottlenecks(framework, trace_file)
        
        # Step 7: Compare configurations
        step_7_compare_configurations(framework, trace_file)
        
        # Step 8: Individual model demo
        step_8_individual_model_demo(framework, trace_file)
        
        print("=" * 70)
        print("WALKTHROUGH COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Read TUTORIAL.md for detailed explanations")
        print("  2. Read SETUP_GUIDE.md for setup instructions")
        print("  3. Read ARCHITECTURE.md for architecture details")
        print("  4. Modify models to match your architecture")
        print("  5. Collect real traces with Tacit")
        print("  6. Calibrate models against real hardware")
        print()
        
    except Exception as e:
        print(f"\n✗ Error during walkthrough: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


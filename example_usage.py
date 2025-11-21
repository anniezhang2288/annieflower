"""
Example usage of the performance modeling framework.
Demonstrates how to use Tacit traces with the analytical models.
"""

from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework
from trace_processor import TacitTraceProcessor


def example_basic_usage():
    """Basic example: process a trace file and get performance predictions."""
    
    # Create default microarchitectural configuration
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
    
    # Process a Tacit trace file
    trace_file = "traces/example_trace.txt"
    results = framework.predict_performance(trace_file, output_file="results.json")
    
    # Print results
    print("Performance Prediction Results:")
    print(f"  IPC: {results['performance_breakdown']['ipc']:.3f}")
    print(f"  Total Cycles: {results['performance_breakdown']['total_cycles']:.0f}")
    print(f"  Stall Cycles: {results['performance_breakdown']['total_stall_cycles']:.0f}")
    
    print("\nComponent Breakdown:")
    for component, metrics in results['model_results'].items():
        print(f"  {component}:")
        print(f"    Stall Cycles: {metrics['stall_cycles']:.2f}")
        print(f"    Throughput: {metrics['throughput']:.3f}")
        print(f"    Utilization: {metrics['utilization']:.3f}")


def example_custom_config():
    """Example: use custom microarchitectural configuration."""
    
    # Custom configuration for a high-performance CPU
    config = MicroarchConfig(
        pipeline_width=6,      # Wider pipeline
        pipeline_depth=16,     # Deeper pipeline
        rob_size=256,          # Larger ROB
        lsq_size=96,          # Larger LSQ
        l1d_size=64 * 1024,    # 64KB L1D
        l2_size=512 * 1024,    # 512KB L2
        l3_size=16 * 1024 * 1024,  # 16MB L3
        bp_size=8192,          # Larger branch predictor
        memory_latency=80      # Lower memory latency
    )
    
    framework = PerformanceModelFramework(config)
    results = framework.process_trace_file("traces/example_trace.txt")
    
    print(f"IPC with custom config: {results['performance_breakdown']['ipc']:.3f}")


def example_compare_configs():
    """Example: compare different microarchitectural configurations."""
    
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
        l1d_size=64 * 1024,    # 2x L1D
        l2_size=512 * 1024,     # 2x L2
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
    
    framework = PerformanceModelFramework()
    comparison = framework.compare_configurations(
        "traces/example_trace.txt",
        [baseline, large_cache, wide_pipeline]
    )
    
    print("Configuration Comparison:")
    for config_name, config_data in comparison.items():
        ipc = config_data['performance']['ipc']
        print(f"  {config_name}: IPC = {ipc:.3f}")


def example_bottleneck_analysis():
    """Example: identify performance bottlenecks."""
    
    framework = PerformanceModelFramework()
    analysis = framework.get_bottleneck_analysis("traces/example_trace.txt")
    
    print("Bottleneck Analysis:")
    print(f"  Primary Bottleneck: {analysis['primary_bottleneck']['component']}")
    print(f"    Stall Cycles: {analysis['primary_bottleneck']['stall_cycles']:.2f}")
    print(f"    Percentage: {analysis['primary_bottleneck']['percentage']:.1f}%")
    
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")


def example_process_trace_directly():
    """Example: process trace data directly (without file)."""
    
    # Create sample trace data (in practice, this comes from Tacit)
    trace_data = [
        {'pc': 0x1000, 'instruction_type': 'load', 'memory_address': 0x2000},
        {'pc': 0x1004, 'instruction_type': 'alu'},
        {'pc': 0x1008, 'instruction_type': 'store', 'memory_address': 0x2004},
        {'pc': 0x100c, 'instruction_type': 'branch', 'branch_taken': True},
        {'pc': 0x2000, 'instruction_type': 'load', 'memory_address': 0x3000},
    ]
    
    framework = PerformanceModelFramework()
    results = framework.process_trace(trace_data)
    
    print(f"IPC: {results['performance_breakdown']['ipc']:.3f}")


if __name__ == "__main__":
    print("Performance Modeling Framework Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    # example_basic_usage()
    # example_custom_config()
    # example_compare_configs()
    # example_bottleneck_analysis()
    # example_process_trace_directly()
    
    print("\nNote: Make sure you have Tacit trace files in the traces/ directory")
    print("or modify the trace file paths in the examples.")


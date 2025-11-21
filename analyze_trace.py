"""
Analyze RISC-V instruction trace and run through performance models.
"""

import json
from pathlib import Path
from trace_parser_riscv import RISCVTraceParser
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework


def analyze_riscv_trace(trace_file: str, output_dir: str = "results"):
    """
    Analyze a RISC-V trace file.
    
    Args:
        trace_file: Path to RISC-V trace file
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("RISC-V Trace Analysis")
    print("=" * 70)
    print(f"\nTrace file: {trace_file}")
    
    # Step 1: Parse the trace
    print("\n[1/4] Parsing trace file...")
    parser = RISCVTraceParser(trace_file)
    trace_data = parser.parse()
    
    print(f"✓ Parsed {len(trace_data):,} instructions")
    
    # Step 2: Get statistics
    print("\n[2/4] Analyzing trace statistics...")
    stats = parser.get_statistics()
    
    print(f"\nTrace Statistics:")
    print(f"  Total Instructions: {stats['total_instructions']:,}")
    print(f"  Unique PCs:         {stats['unique_pcs']:,}")
    print(f"  Estimated Cycles:   {stats['cycles']:,}")
    print(f"  Branches:           {stats['branches']:,}")
    print(f"  Loads:              {stats['loads']:,}")
    print(f"  Stores:             {stats['stores']:,}")
    
    print(f"\nInstruction Type Distribution:")
    for inst_type, count in sorted(stats['instruction_types'].items(), 
                                   key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_instructions']) * 100
        print(f"  {inst_type:12s}: {count:8,} ({percentage:5.1f}%)")
    
    # Step 3: Convert to standard format and save
    print("\n[3/4] Converting to standard format...")
    standard_trace = []
    for entry in trace_data:
        standard_entry = {
            'pc': entry['pc'],
            'instruction_type': entry['instruction_type'],
            'cycle': entry.get('cycle', 0)
        }
        
        if 'memory_address' in entry:
            standard_entry['memory_address'] = entry['memory_address']
        if 'branch_taken' in entry:
            standard_entry['branch_taken'] = entry['branch_taken']
        
        standard_trace.append(standard_entry)
    
    # Save converted trace
    Path(output_dir).mkdir(exist_ok=True)
    converted_trace_file = Path(output_dir) / "converted_trace.json"
    with open(converted_trace_file, 'w') as f:
        json.dump(standard_trace, f, indent=2)
    
    print(f"✓ Saved converted trace: {converted_trace_file}")
    print(f"  (First 1000 instructions saved for performance)")
    
    # Step 4: Run performance models
    print("\n[4/4] Running performance models...")
    print("  (Using sample of trace for faster analysis)")
    
    # Use a sample for faster analysis (or use full trace if you want)
    sample_size = min(10000, len(standard_trace))
    sample_trace = standard_trace[:sample_size]
    
    print(f"  Analyzing {sample_size:,} instructions...")
    
    # Configure microarchitecture (RISC-V typical)
    config = MicroarchConfig(
        pipeline_width=4,
        pipeline_depth=14,
        rob_size=192,
        lsq_size=72,
        l1d_size=32 * 1024,    # 32KB
        l2_size=256 * 1024,     # 256KB
        l3_size=8 * 1024 * 1024,  # 8MB
        bp_size=4096,
        memory_latency=100
    )
    
    framework = PerformanceModelFramework(config)
    results = framework.process_trace(sample_trace)
    
    # Print results
    print(f"\n{'='*70}")
    print("Performance Prediction Results")
    print(f"{'='*70}")
    
    perf = results['performance_breakdown']
    print(f"\nOverall Performance:")
    print(f"  Instructions Analyzed: {sample_size:,}")
    print(f"  Total Cycles:          {perf['total_cycles']:,.0f}")
    print(f"  Total Stall Cycles:    {perf['total_stall_cycles']:,.2f}")
    print(f"  IPC:                   {perf['ipc']:.3f}")
    print(f"  Overall Throughput:    {perf['overall_throughput']:.3f}")
    
    print(f"\nStall Cycle Breakdown:")
    for component, stalls in sorted(perf['stall_breakdown'].items(), 
                                    key=lambda x: x[1], reverse=True):
        percentage = (stalls / perf['total_stall_cycles'] * 100) if perf['total_stall_cycles'] > 0 else 0
        print(f"  {component:25s}: {stalls:8.2f} cycles ({percentage:5.1f}%)")
    
    print(f"\nComponent Details:")
    for component, metrics in results['model_results'].items():
        print(f"\n  {component}:")
        print(f"    Stall Cycles:  {metrics['stall_cycles']:8.2f}")
        print(f"    Throughput:    {metrics['throughput']:.3f}")
        print(f"    Utilization:   {metrics['utilization']:.3f}")
        
        # Show key additional metrics
        if metrics['additional_metrics']:
            key_metrics = {}
            for key, value in metrics['additional_metrics'].items():
                if isinstance(value, (int, float)) and key in [
                    'prediction_accuracy', 'l1_hit_rate', 'l2_hit_rate', 
                    'l3_hit_rate', 'avg_rob_occupancy'
                ]:
                    key_metrics[key] = value
            
            if key_metrics:
                print(f"    Key Metrics:")
                for key, value in key_metrics.items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.3f}")
                    else:
                        print(f"      {key}: {value}")
    
    # Bottleneck analysis
    print(f"\n{'='*70}")
    print("Bottleneck Analysis")
    print(f"{'='*70}")
    
    # Manual bottleneck analysis
    stall_breakdown = perf['stall_breakdown']
    if stall_breakdown:
        max_stall = max(stall_breakdown.items(), key=lambda x: x[1])
        print(f"\nPrimary Bottleneck: {max_stall[0]}")
        print(f"  Stall Cycles: {max_stall[1]:.2f}")
        if perf['total_stall_cycles'] > 0:
            print(f"  Percentage: {(max_stall[1] / perf['total_stall_cycles'] * 100):.1f}%")
    
    # Save results
    results_file = Path(output_dir) / "performance_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'trace_statistics': stats,
            'sample_size': sample_size,
            'performance_results': results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Estimate for full trace
    if len(standard_trace) > sample_size:
        print(f"\n{'='*70}")
        print("Full Trace Estimation")
        print(f"{'='*70}")
        print(f"\nSample analyzed: {sample_size:,} instructions")
        print(f"Full trace:      {len(standard_trace):,} instructions")
        print(f"Scale factor:    {len(standard_trace) / sample_size:.2f}x")
        
        estimated_cycles = perf['total_cycles'] * (len(standard_trace) / sample_size)
        estimated_ipc = perf['ipc']  # IPC should be similar
        print(f"\nEstimated for full trace:")
        print(f"  Estimated Cycles: {estimated_cycles:,.0f}")
        print(f"  Estimated IPC:    {estimated_ipc:.3f}")
    
    return results, stats


def main():
    """Main analysis function."""
    trace_file = "data/dhrystone.riscv.out.out"
    
    if not Path(trace_file).exists():
        print(f"Error: Trace file not found: {trace_file}")
        return
    
    try:
        results, stats = analyze_riscv_trace(trace_file)
        print(f"\n{'='*70}")
        print("Analysis Complete!")
        print(f"{'='*70}")
        print("\nNext steps:")
        print("  1. Review results in results/performance_results.json")
        print("  2. Adjust microarchitectural parameters in analyze_trace.py")
        print("  3. Run full trace analysis (modify sample_size)")
        print("  4. Compare different configurations")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


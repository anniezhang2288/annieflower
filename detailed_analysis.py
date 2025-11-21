"""
Detailed analysis of the dhrystone trace with visualizations and insights.
"""

import json
from pathlib import Path
from trace_parser_riscv import RISCVTraceParser
from analytical import MicroarchConfig
from performance_model import PerformanceModelFramework


def detailed_analysis(trace_file: str = "data/dhrystone.riscv.out.out"):
    """Perform detailed analysis with insights."""
    
    print("=" * 70)
    print("DETAILED TRACE ANALYSIS: Dhrystone Benchmark")
    print("=" * 70)
    
    # Parse trace
    parser = RISCVTraceParser(trace_file)
    trace_data = parser.parse()
    stats = parser.get_statistics()
    
    print(f"\n📊 TRACE OVERVIEW")
    print(f"{'─' * 70}")
    print(f"Total Instructions:     {stats['total_instructions']:,}")
    print(f"Unique Program Counters: {stats['unique_pcs']:,}")
    print(f"Code Reuse Ratio:        {stats['total_instructions'] / stats['unique_pcs']:.1f}x")
    print(f"Estimated Execution Time: {stats['cycles']:,} cycles")
    
    print(f"\n📈 INSTRUCTION MIX")
    print(f"{'─' * 70}")
    total = stats['total_instructions']
    for inst_type, count in sorted(stats['instruction_types'].items(), 
                                   key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"{inst_type:12s}: {count:8,} ({percentage:5.1f}%) {bar}")
    
    # Memory access analysis
    print(f"\n💾 MEMORY ACCESS PATTERNS")
    print(f"{'─' * 70}")
    loads = stats['loads']
    stores = stats['stores']
    total_mem = loads + stores
    load_ratio = loads / total_mem if total_mem > 0 else 0
    print(f"Total Memory Operations: {total_mem:,}")
    print(f"  Loads:  {loads:,} ({load_ratio*100:.1f}%)")
    print(f"  Stores: {stores:,} ({(1-load_ratio)*100:.1f}%)")
    print(f"Load/Store Ratio: {loads/stores:.2f}:1" if stores > 0 else "N/A")
    
    # Branch analysis
    print(f"\n🔀 BRANCH CHARACTERISTICS")
    print(f"{'─' * 70}")
    branches = stats['branches']
    branch_frequency = branches / total * 100
    print(f"Branch Instructions:     {branches:,}")
    print(f"Branch Frequency:        {branch_frequency:.1f}%")
    print(f"Average Instructions per Branch: {total / branches:.1f}" if branches > 0 else "N/A")
    
    # Performance modeling
    print(f"\n⚙️  PERFORMANCE MODELING")
    print(f"{'─' * 70}")
    
    # Use larger sample for better accuracy
    sample_size = min(50000, len(trace_data))
    sample_trace = []
    
    for entry in trace_data[:sample_size]:
        standard_entry = {
            'pc': entry['pc'],
            'instruction_type': entry['instruction_type'],
            'cycle': entry.get('cycle', 0)
        }
        if 'memory_address' in entry:
            standard_entry['memory_address'] = entry['memory_address']
        if 'branch_taken' in entry:
            standard_entry['branch_taken'] = entry['branch_taken']
        sample_trace.append(standard_entry)
    
    print(f"Analyzing {sample_size:,} instruction sample...")
    
    # Baseline configuration
    baseline_config = MicroarchConfig(
        pipeline_width=4,
        pipeline_depth=14,
        rob_size=192,
        lsq_size=72,
        l1d_size=32 * 1024,
        l2_size=256 * 1024,
        l3_size=8 * 1024 * 1024,
        bp_size=4096,
        memory_latency=100
    )
    
    framework = PerformanceModelFramework(baseline_config)
    results = framework.process_trace(sample_trace)
    
    perf = results['performance_breakdown']
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'─' * 70}")
    print(f"IPC (Instructions Per Cycle):     {perf['ipc']:.3f}")
    print(f"Total Cycles:                      {perf['total_cycles']:,.0f}")
    print(f"Stall Cycles:                      {perf['total_stall_cycles']:,.0f}")
    print(f"Stall Percentage:                  {(perf['total_stall_cycles']/perf['total_cycles']*100):.1f}%")
    print(f"Effective Throughput:             {perf['overall_throughput']:.3f}")
    
    print(f"\n🔍 COMPONENT ANALYSIS")
    print(f"{'─' * 70}")
    
    # Sort by stall cycles
    stall_breakdown = sorted(perf['stall_breakdown'].items(), 
                            key=lambda x: x[1], reverse=True)
    
    for component, stalls in stall_breakdown:
        if stalls > 0:
            percentage = (stalls / perf['total_stall_cycles'] * 100) if perf['total_stall_cycles'] > 0 else 0
            metrics = results['model_results'][component]
            
            print(f"\n{component}:")
            print(f"  Stall Cycles:  {stalls:8.2f} ({percentage:5.1f}%)")
            print(f"  Throughput:   {metrics['throughput']:.3f}")
            print(f"  Utilization:   {metrics['utilization']:.3f}")
            
            # Component-specific insights
            add_metrics = metrics.get('additional_metrics', {})
            
            if component == 'BranchPredictorModel':
                accuracy = add_metrics.get('prediction_accuracy', 0)
                mispred_rate = add_metrics.get('misprediction_rate', 0)
                print(f"  Prediction Accuracy: {accuracy:.1%}")
                print(f"  Misprediction Rate:  {mispred_rate:.1%}")
                if mispred_rate > 0.05:
                    print(f"  ⚠️  High misprediction rate - consider larger predictor")
            
            elif component == 'CacheModel':
                l1_hit = add_metrics.get('l1_hit_rate', 0)
                l2_hit = add_metrics.get('l2_hit_rate', 0)
                l3_hit = add_metrics.get('l3_hit_rate', 0)
                mem_rate = add_metrics.get('memory_access_rate', 0)
                print(f"  L1 Hit Rate:         {l1_hit:.1%}")
                print(f"  L2 Hit Rate:         {l2_hit:.1%}")
                print(f"  L3 Hit Rate:         {l3_hit:.1%}")
                print(f"  Memory Access Rate:  {mem_rate:.1%}")
                if l1_hit < 0.90:
                    print(f"  ⚠️  Low L1 hit rate - consider larger L1 cache")
                if mem_rate > 0.10:
                    print(f"  ⚠️  High memory access rate - consider larger L3 cache")
            
            elif component == 'LoadStoreQueueModel':
                load_count = add_metrics.get('load_count', 0)
                store_count = add_metrics.get('store_count', 0)
                dep_stalls = add_metrics.get('dependency_stalls', 0)
                cap_stalls = add_metrics.get('capacity_stalls', 0)
                print(f"  Loads:                {load_count:,}")
                print(f"  Stores:               {store_count:,}")
                print(f"  Dependency Stalls:    {dep_stalls:.2f}")
                print(f"  Capacity Stalls:      {cap_stalls:.2f}")
                if cap_stalls > dep_stalls:
                    print(f"  ⚠️  LSQ capacity limited - consider larger LSQ")
                else:
                    print(f"  ⚠️  Memory dependencies limiting - consider better forwarding")
            
            elif component == 'ReorderBufferModel':
                occupancy = add_metrics.get('avg_rob_occupancy', 0)
                max_occ = add_metrics.get('max_rob_occupancy', 0)
                print(f"  Avg ROB Occupancy:    {occupancy:.1f}")
                print(f"  Max ROB Occupancy:    {max_occ:.1f}")
                if occupancy > baseline_config.rob_size * 0.8:
                    print(f"  ⚠️  High ROB occupancy - consider larger ROB")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS")
    print(f"{'─' * 70}")
    
    recommendations = []
    
    # Find primary bottleneck
    primary_bottleneck = max(stall_breakdown, key=lambda x: x[1])
    
    if primary_bottleneck[0] == 'LoadStoreQueueModel':
        recommendations.append("🔧 Primary bottleneck is LSQ - consider:")
        recommendations.append("   • Increase LSQ size (current: 72 entries)")
        recommendations.append("   • Improve store-to-load forwarding")
        recommendations.append("   • Better memory disambiguation")
    
    elif primary_bottleneck[0] == 'BranchPredictorModel':
        recommendations.append("🔧 Primary bottleneck is branch predictor - consider:")
        recommendations.append("   • Increase predictor size (current: 4K entries)")
        recommendations.append("   • Use more sophisticated predictor (gshare, perceptron)")
        recommendations.append("   • Improve branch target buffer")
    
    elif primary_bottleneck[0] == 'CacheModel':
        recommendations.append("🔧 Primary bottleneck is cache hierarchy - consider:")
        recommendations.append("   • Increase L1 cache size (current: 32KB)")
        recommendations.append("   • Increase L2/L3 cache sizes")
        recommendations.append("   • Improve prefetching")
    
    # Add general recommendations
    if perf['ipc'] < 1.0:
        recommendations.append(f"\n📈 Overall IPC is {perf['ipc']:.3f} - potential improvements:")
        recommendations.append("   • Wider pipeline (current: 4-wide)")
        recommendations.append("   • More execution units")
        recommendations.append("   • Better instruction scheduling")
    
    for rec in recommendations:
        print(rec)
    
    # Configuration comparison
    print(f"\n🔬 CONFIGURATION SENSITIVITY")
    print(f"{'─' * 70}")
    
    configs = {
        'Baseline': baseline_config,
        'Wider Pipeline': MicroarchConfig(
            pipeline_width=6, rob_size=256, lsq_size=96,
            l1d_size=32*1024, l2_size=256*1024, l3_size=8*1024*1024
        ),
        'Larger Caches': MicroarchConfig(
            pipeline_width=4, rob_size=192, lsq_size=72,
            l1d_size=64*1024, l2_size=512*1024, l3_size=16*1024*1024
        ),
        'Larger Buffers': MicroarchConfig(
            pipeline_width=4, rob_size=256, lsq_size=96,
            l1d_size=32*1024, l2_size=256*1024, l3_size=8*1024*1024
        ),
    }
    
    print(f"\n{'Configuration':<20} {'IPC':<10} {'Improvement':<15}")
    print(f"{'─' * 45}")
    
    baseline_ipc = perf['ipc']
    
    for name, config in configs.items():
        if name == 'Baseline':
            print(f"{name:<20} {baseline_ipc:<10.3f} {'baseline':<15}")
        else:
            test_framework = PerformanceModelFramework(config)
            test_results = test_framework.process_trace(sample_trace)
            test_ipc = test_results['performance_breakdown']['ipc']
            improvement = ((test_ipc - baseline_ipc) / baseline_ipc) * 100
            print(f"{name:<20} {test_ipc:<10.3f} {improvement:+.1f}%")
    
    # Save detailed report
    report = {
        'trace_statistics': stats,
        'performance_results': results,
        'sample_size': sample_size,
        'recommendations': recommendations
    }
    
    Path("results").mkdir(exist_ok=True)
    with open("results/detailed_analysis.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Detailed analysis saved to: results/detailed_analysis.json")
    
    return results, stats


if __name__ == "__main__":
    detailed_analysis()


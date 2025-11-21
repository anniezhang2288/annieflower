"""
Main performance modeling framework.
Combines all analytical models to predict overall CPU performance.
Similar to Concorde's hybrid analytical-ML approach.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from analytical import (
    MicroarchConfig,
    BranchPredictorModel,
    CacheModel,
    ReorderBufferModel,
    LoadStoreQueueModel,
    MemoryModel,
    PerformanceMetrics
)
from trace_processor import TacitTraceProcessor


class PerformanceModelFramework:
    """
    Main framework for performance modeling.
    
    Combines multiple analytical models to predict CPU performance
    from Tacit instruction traces.
    """
    
    def __init__(self, config: Optional[MicroarchConfig] = None):
        """
        Initialize the performance modeling framework.
        
        Args:
            config: Microarchitectural configuration. If None, uses defaults.
        """
        self.config = config or MicroarchConfig()
        
        # Initialize all analytical models
        self.bp_model = BranchPredictorModel(self.config)
        self.cache_model = CacheModel(self.config)
        self.rob_model = ReorderBufferModel(self.config)
        self.lsq_model = LoadStoreQueueModel(self.config)
        self.memory_model = MemoryModel(self.config)
        
        # Trace processor
        self.trace_processor = TacitTraceProcessor()
        
        # All models
        self.models = [
            self.bp_model,
            self.cache_model,
            self.rob_model,
            self.lsq_model,
            self.memory_model
        ]
    
    def process_trace_file(self, trace_file: str) -> Dict[str, Any]:
        """
        Process a Tacit trace file and compute performance predictions.
        
        Args:
            trace_file: Path to Tacit instruction trace file
            
        Returns:
            Dictionary containing performance predictions and metrics
        """
        # Load and process trace
        trace_data = self.trace_processor.load_trace(trace_file)
        
        return self.process_trace(trace_data)
    
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process instruction trace data and compute performance predictions.
        
        Args:
            trace_data: List of instruction trace entries
            
        Returns:
            Dictionary containing performance predictions and metrics
        """
        if not trace_data:
            return {
                'error': 'Empty trace data',
                'metrics': {}
            }
        
        # Process each model
        model_results = {}
        total_stall_cycles = 0.0
        total_latency = 0.0
        min_throughput = 1.0
        
        for model in self.models:
            metrics = model.process_trace(trace_data)
            model_results[model.name] = {
                'latency': metrics.latency,
                'throughput': metrics.throughput,
                'utilization': metrics.utilization,
                'stall_cycles': metrics.stall_cycles,
                'additional_metrics': metrics.additional_metrics
            }
            
            total_stall_cycles += metrics.stall_cycles
            total_latency += metrics.latency
            min_throughput = min(min_throughput, metrics.throughput)
        
        # Compute overall performance metrics
        total_instructions = len(trace_data)
        total_cycles = total_instructions + total_stall_cycles
        
        # Overall IPC (Instructions Per Cycle)
        ipc = total_instructions / total_cycles if total_cycles > 0 else 0.0
        
        # Overall throughput (normalized)
        overall_throughput = min_throughput
        
        # Performance breakdown
        performance_breakdown = {
            'total_instructions': total_instructions,
            'total_cycles': total_cycles,
            'total_stall_cycles': total_stall_cycles,
            'ipc': ipc,
            'overall_throughput': overall_throughput,
            'stall_breakdown': {
                model.name: model_results[model.name]['stall_cycles']
                for model in self.models
            }
        }
        
        # Trace statistics
        trace_stats = self.trace_processor.get_instruction_statistics(trace_data)
        
        return {
            'performance_breakdown': performance_breakdown,
            'model_results': model_results,
            'trace_statistics': trace_stats,
            'config': {
                'pipeline_width': self.config.pipeline_width,
                'pipeline_depth': self.config.pipeline_depth,
                'rob_size': self.config.rob_size,
                'lsq_size': self.config.lsq_size,
                'l1d_size': self.config.l1d_size,
                'l2_size': self.config.l2_size,
                'l3_size': self.config.l3_size,
            }
        }
    
    def predict_performance(self, trace_file: str, 
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict performance from trace file and optionally save results.
        
        Args:
            trace_file: Path to Tacit instruction trace file
            output_file: Optional path to save results JSON
            
        Returns:
            Performance prediction results
        """
        results = self.process_trace_file(trace_file)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def compare_configurations(self, trace_file: str, 
                              configs: List[MicroarchConfig]) -> Dict[str, Any]:
        """
        Compare performance across different microarchitectural configurations.
        
        Args:
            trace_file: Path to Tacit instruction trace file
            configs: List of microarchitectural configurations to compare
            
        Returns:
            Comparison results for each configuration
        """
        trace_data = self.trace_processor.load_trace(trace_file)
        
        comparison_results = {}
        
        for i, config in enumerate(configs):
            # Create new framework with this configuration
            framework = PerformanceModelFramework(config)
            results = framework.process_trace(trace_data)
            
            comparison_results[f'config_{i}'] = {
                'config': {
                    'pipeline_width': config.pipeline_width,
                    'rob_size': config.rob_size,
                    'lsq_size': config.lsq_size,
                    'l1d_size': config.l1d_size,
                    'l2_size': config.l2_size,
                    'l3_size': config.l3_size,
                },
                'performance': results['performance_breakdown']
            }
        
        return comparison_results
    
    def get_bottleneck_analysis(self, trace_file: str) -> Dict[str, Any]:
        """
        Identify performance bottlenecks from trace analysis.
        
        Args:
            trace_file: Path to Tacit instruction trace file
            
        Returns:
            Bottleneck analysis results
        """
        results = self.process_trace_file(trace_file)
        
        # Find component with highest stall cycles
        stall_breakdown = results['performance_breakdown']['stall_breakdown']
        max_stall_component = max(stall_breakdown.items(), key=lambda x: x[1])
        
        # Find component with lowest throughput
        model_results = results['model_results']
        min_throughput_component = min(
            model_results.items(),
            key=lambda x: x[1]['throughput']
        )
        
        # Find component with highest utilization
        max_utilization_component = max(
            model_results.items(),
            key=lambda x: x[1]['utilization']
        )
        
        return {
            'primary_bottleneck': {
                'component': max_stall_component[0],
                'stall_cycles': max_stall_component[1],
                'percentage': (max_stall_component[1] / 
                              results['performance_breakdown']['total_stall_cycles'] * 100
                              if results['performance_breakdown']['total_stall_cycles'] > 0 else 0)
            },
            'throughput_bottleneck': {
                'component': min_throughput_component[0],
                'throughput': min_throughput_component[1]['throughput']
            },
            'utilization_bottleneck': {
                'component': max_utilization_component[0],
                'utilization': max_utilization_component[1]['utilization']
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        stall_breakdown = results['performance_breakdown']['stall_breakdown']
        model_results = results['model_results']
        
        # Check branch predictor
        if stall_breakdown.get('BranchPredictorModel', 0) > 0:
            bp_stalls = stall_breakdown['BranchPredictorModel']
            total_stalls = results['performance_breakdown']['total_stall_cycles']
            if bp_stalls / total_stalls > 0.2:
                recommendations.append(
                    "Consider increasing branch predictor size or improving prediction algorithm"
                )
        
        # Check cache
        cache_result = model_results.get('CacheModel', {})
        if cache_result.get('additional_metrics', {}).get('memory_access_rate', 0) > 0.1:
            recommendations.append(
                "High memory access rate detected. Consider increasing cache sizes or improving prefetching"
            )
        
        # Check ROB
        if stall_breakdown.get('ReorderBufferModel', 0) > 0:
            recommendations.append(
                "ROB capacity may be limiting performance. Consider increasing ROB size"
            )
        
        # Check LSQ
        if stall_breakdown.get('LoadStoreQueueModel', 0) > 0:
            recommendations.append(
                "LSQ capacity or memory dependencies may be limiting performance"
            )
        
        return recommendations


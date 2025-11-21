"""
Branch Predictor Performance Model.
Models the performance impact of branch prediction accuracy and latency.
"""

from typing import List, Dict, Any
import numpy as np
from .base_model import BasePerformanceModel, PerformanceMetrics, MicroarchConfig


class BranchPredictorModel(BasePerformanceModel):
    """
    Analytical model for branch predictor performance.
    
    Models:
    - Branch prediction accuracy
    - Misprediction penalty
    - Branch predictor latency
    - Branch target buffer (BTB) hits/misses
    """
    
    def __init__(self, config: MicroarchConfig):
        super().__init__(config)
        self.misprediction_penalty = config.pipeline_depth  # Flush pipeline on mispredict
        
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Process trace and compute branch predictor metrics."""
        branches = self._extract_branches(trace_data)
        
        if not branches:
            return PerformanceMetrics(
                latency=0.0,
                throughput=1.0,
                utilization=0.0,
                stall_cycles=0.0
            )
        
        # Calculate prediction accuracy
        accuracy = self._estimate_accuracy(branches)
        misprediction_rate = 1.0 - accuracy
        
        # Calculate stall cycles due to mispredictions
        stall_cycles = self._calculate_misprediction_stalls(branches, misprediction_rate)
        
        # Calculate latency
        latency = self._estimate_bp_latency(branches)
        
        # Calculate throughput impact
        throughput = self._estimate_throughput_impact(branches, misprediction_rate)
        
        # Utilization (how often BP is accessed)
        utilization = len(branches) / len(trace_data) if trace_data else 0.0
        
        return PerformanceMetrics(
            latency=latency,
            throughput=throughput,
            utilization=utilization,
            stall_cycles=stall_cycles,
            additional_metrics={
                'prediction_accuracy': accuracy,
                'misprediction_rate': misprediction_rate,
                'branch_count': len(branches),
                'misprediction_count': int(len(branches) * misprediction_rate)
            }
        )
    
    def estimate_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate branch predictor latency contribution."""
        branches = self._extract_branches(trace_data)
        if not branches:
            return 0.0
        
        # Base latency per branch lookup
        lookup_latency = 1.0  # Typically 1 cycle
        
        # Add latency for BTB misses (if applicable)
        btb_miss_rate = self._estimate_btb_miss_rate(branches)
        btb_miss_latency = btb_miss_rate * 2.0  # Additional cycles for BTB miss
        
        return len(branches) * (lookup_latency + btb_miss_latency)
    
    def estimate_throughput(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate throughput impact from branch predictor."""
        branches = self._extract_branches(trace_data)
        if not branches:
            return 1.0
        
        misprediction_rate = 1.0 - self._estimate_accuracy(branches)
        
        # Throughput reduction due to pipeline flushes
        # Each misprediction flushes the pipeline
        throughput_penalty = misprediction_rate * (self.misprediction_penalty / len(trace_data))
        
        return max(0.0, 1.0 - throughput_penalty)
    
    def get_stall_cycles(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate stall cycles due to branch mispredictions."""
        branches = self._extract_branches(trace_data)
        return self._calculate_misprediction_stalls(branches, 
                                                   1.0 - self._estimate_accuracy(branches))
    
    def get_utilization(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate branch predictor utilization."""
        branches = self._extract_branches(trace_data)
        return len(branches) / len(trace_data) if trace_data else 0.0
    
    def _extract_branches(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract branch instructions from trace."""
        branch_types = ['branch', 'jump', 'call', 'ret', 'conditional']
        return [entry for entry in trace_data 
                if entry.get('instruction_type') in branch_types]
    
    def _estimate_accuracy(self, branches: List[Dict[str, Any]]) -> float:
        """
        Estimate branch prediction accuracy.
        
        Uses analytical model based on:
        - Predictor size and associativity
        - Branch pattern characteristics
        - Historical accuracy data (if available)
        """
        if not branches:
            return 1.0
        
        # Base accuracy based on predictor configuration
        # Larger predictors generally have higher accuracy
        base_accuracy = 0.95  # Typical for modern predictors
        
        # Adjust based on predictor size
        size_factor = min(1.0, np.log2(self.config.bp_size) / 12.0)  # Normalize to 4KB
        accuracy = base_accuracy * (0.8 + 0.2 * size_factor)
        
        # Pattern-based adjustments (simplified)
        # Real implementation would analyze branch patterns
        taken_rate = sum(1 for b in branches if b.get('branch_taken', False)) / len(branches)
        
        # Highly biased branches are easier to predict
        if taken_rate > 0.9 or taken_rate < 0.1:
            accuracy += 0.03
        elif 0.4 < taken_rate < 0.6:
            accuracy -= 0.02  # Harder to predict
        
        return min(0.99, max(0.85, accuracy))
    
    def _estimate_btb_miss_rate(self, branches: List[Dict[str, Any]]) -> float:
        """Estimate Branch Target Buffer miss rate."""
        if not branches:
            return 0.0
        
        # Simplified model: BTB miss rate depends on BTB size
        # Assume BTB size is similar to predictor size
        btb_entries = self.config.bp_size // 4  # Rough estimate
        unique_targets = len(set(b.get('pc', 0) for b in branches))
        
        # If we have more unique targets than BTB entries, expect misses
        if unique_targets > btb_entries:
            return min(0.1, (unique_targets - btb_entries) / len(branches))
        else:
            return 0.01  # Low miss rate
    
    def _calculate_misprediction_stalls(self, branches: List[Dict[str, Any]], 
                                       misprediction_rate: float) -> float:
        """Calculate total stall cycles from mispredictions."""
        if not branches:
            return 0.0
        
        mispredictions = len(branches) * misprediction_rate
        return mispredictions * self.misprediction_penalty
    
    def _estimate_bp_latency(self, branches: List[Dict[str, Any]]) -> float:
        """Estimate total branch predictor latency."""
        if not branches:
            return 0.0
        
        lookup_latency = 1.0  # Cycles per lookup
        btb_miss_rate = self._estimate_btb_miss_rate(branches)
        btb_miss_penalty = 2.0  # Additional cycles for BTB miss
        
        return len(branches) * (lookup_latency + btb_miss_rate * btb_miss_penalty)
    
    def _estimate_throughput_impact(self, branches: List[Dict[str, Any]], 
                                   misprediction_rate: float) -> float:
        """Estimate throughput impact from branch predictor."""
        if not branches:
            return 1.0
        
        # Throughput is reduced by pipeline flushes from mispredictions
        total_cycles = len(branches)  # Simplified
        stall_cycles = self._calculate_misprediction_stalls(branches, misprediction_rate)
        
        if total_cycles == 0:
            return 1.0
        
        return 1.0 - (stall_cycles / (total_cycles + stall_cycles))


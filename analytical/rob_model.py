"""
Reorder Buffer (ROB) Performance Model.
Models the performance impact of instruction retirement and ROB capacity.
"""

from typing import List, Dict, Any
import numpy as np
from .base_model import BasePerformanceModel, PerformanceMetrics, MicroarchConfig


class ReorderBufferModel(BasePerformanceModel):
    """
    Analytical model for Reorder Buffer performance.
    
    Models:
    - ROB capacity constraints
    - Instruction retirement rate
    - ROB stalls due to full buffer
    - Dependency chain impacts
    """
    
    def __init__(self, config: MicroarchConfig):
        super().__init__(config)
        self.rob_size = config.rob_size
        self.retirement_width = config.pipeline_width  # Instructions retired per cycle
    
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Process trace and compute ROB performance metrics."""
        if not trace_data:
            return PerformanceMetrics(
                latency=0.0,
                throughput=1.0,
                utilization=0.0,
                stall_cycles=0.0
            )
        
        # Analyze ROB occupancy
        rob_occupancy = self._estimate_rob_occupancy(trace_data)
        
        # Calculate stalls due to full ROB
        stall_cycles = self._calculate_rob_stalls(trace_data, rob_occupancy)
        
        # Calculate retirement latency
        latency = self._estimate_retirement_latency(trace_data)
        
        # Calculate throughput impact
        throughput = self._estimate_throughput_impact(trace_data, stall_cycles)
        
        # Utilization (average ROB occupancy)
        utilization = min(1.0, rob_occupancy / self.rob_size)
        
        return PerformanceMetrics(
            latency=latency,
            throughput=throughput,
            utilization=utilization,
            stall_cycles=stall_cycles,
            additional_metrics={
                'avg_rob_occupancy': rob_occupancy,
                'max_rob_occupancy': min(self.rob_size, rob_occupancy * 1.2),
                'rob_full_cycles': stall_cycles / self.rob_size if self.rob_size > 0 else 0.0,
                'retirement_rate': self.retirement_width
            }
        )
    
    def estimate_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate ROB retirement latency."""
        return self._estimate_retirement_latency(trace_data)
    
    def estimate_throughput(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate throughput impact from ROB constraints."""
        rob_occupancy = self._estimate_rob_occupancy(trace_data)
        stall_cycles = self._calculate_rob_stalls(trace_data, rob_occupancy)
        return self._estimate_throughput_impact(trace_data, stall_cycles)
    
    def get_stall_cycles(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate stall cycles due to ROB capacity."""
        rob_occupancy = self._estimate_rob_occupancy(trace_data)
        return self._calculate_rob_stalls(trace_data, rob_occupancy)
    
    def get_utilization(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate ROB utilization."""
        rob_occupancy = self._estimate_rob_occupancy(trace_data)
        return min(1.0, rob_occupancy / self.rob_size)
    
    def _estimate_rob_occupancy(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Estimate average ROB occupancy.
        
        Based on:
        - Instruction issue rate
        - Instruction completion rate (longest latency instruction)
        - Dependency chains
        """
        if not trace_data:
            return 0.0
        
        # Estimate average instruction latency
        avg_latency = self._estimate_avg_instruction_latency(trace_data)
        
        # Issue rate (limited by pipeline width)
        issue_rate = min(self.config.pipeline_width, len(trace_data) / max(1, len(trace_data)))
        
        # ROB occupancy = issue_rate * avg_latency (Little's Law)
        occupancy = issue_rate * avg_latency
        
        # Account for dependency chains
        dependency_factor = self._estimate_dependency_impact(trace_data)
        occupancy *= dependency_factor
        
        return min(self.rob_size, occupancy)
    
    def _estimate_avg_instruction_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate average instruction completion latency."""
        if not trace_data:
            return 1.0
        
        # Base latency by instruction type
        latencies = {
            'alu': 1.0,
            'mul': 3.0,
            'div': 10.0,
            'fpu': 3.0,
            'load': 4.0,  # Assuming L1 hit
            'store': 1.0,
            'branch': 1.0,
        }
        
        total_latency = 0.0
        for entry in trace_data:
            inst_type = entry.get('instruction_type', 'alu')
            latency = latencies.get(inst_type, 2.0)
            total_latency += latency
        
        return total_latency / len(trace_data)
    
    def _estimate_dependency_impact(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Estimate impact of dependency chains on ROB occupancy.
        Longer dependency chains increase ROB occupancy.
        """
        if not trace_data:
            return 1.0
        
        # Count dependencies
        dependency_count = sum(len(entry.get('dependencies', [])) 
                             for entry in trace_data)
        
        # Average dependencies per instruction
        avg_deps = dependency_count / len(trace_data)
        
        # More dependencies -> higher ROB occupancy
        # Factor ranges from 1.0 (no deps) to 1.5 (many deps)
        factor = 1.0 + min(0.5, avg_deps * 0.1)
        
        return factor
    
    def _calculate_rob_stalls(self, trace_data: List[Dict[str, Any]], 
                            rob_occupancy: float) -> float:
        """
        Calculate stall cycles when ROB is full.
        
        Stalls occur when:
        - ROB occupancy reaches capacity
        - Retirement rate < issue rate
        """
        if rob_occupancy < self.rob_size:
            return 0.0
        
        # Estimate cycles where ROB is full
        overflow = rob_occupancy - self.rob_size
        stall_probability = overflow / self.rob_size
        
        # Stalls occur when we can't issue new instructions
        total_cycles = len(trace_data)
        stall_cycles = total_cycles * stall_probability
        
        return stall_cycles
    
    def _estimate_retirement_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Estimate latency for instruction retirement.
        
        Retirement is limited by:
        - Retirement width (instructions per cycle)
        - Instruction completion order
        """
        if not trace_data:
            return 0.0
        
        # Minimum cycles to retire all instructions
        min_retirement_cycles = len(trace_data) / self.retirement_width
        
        # Add latency for out-of-order completion
        avg_latency = self._estimate_avg_instruction_latency(trace_data)
        completion_cycles = avg_latency
        
        # Total latency is max of retirement and completion
        return max(min_retirement_cycles, completion_cycles)
    
    def _estimate_throughput_impact(self, trace_data: List[Dict[str, Any]], 
                                   stall_cycles: float) -> float:
        """Estimate throughput impact from ROB stalls."""
        if not trace_data:
            return 1.0
        
        total_cycles = len(trace_data) + stall_cycles
        return len(trace_data) / total_cycles if total_cycles > 0 else 1.0


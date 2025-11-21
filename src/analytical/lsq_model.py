"""
Load-Store Queue (LSQ) Performance Model.
Models the performance impact of memory instruction scheduling and dependencies.
"""

from typing import List, Dict, Any
import numpy as np
from .base_model import BasePerformanceModel, PerformanceMetrics, MicroarchConfig


class LoadStoreQueueModel(BasePerformanceModel):
    """
    Analytical model for Load-Store Queue performance.
    
    Models:
    - LSQ capacity constraints
    - Memory dependency stalls
    - Store-to-load forwarding
    - Memory disambiguation
    """
    
    def __init__(self, config: MicroarchConfig):
        super().__init__(config)
        self.lsq_size = config.lsq_size
        self.load_store_units = config.load_store_units
    
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Process trace and compute LSQ performance metrics."""
        memory_ops = self._extract_memory_operations(trace_data)
        
        if not memory_ops:
            return PerformanceMetrics(
                latency=0.0,
                throughput=1.0,
                utilization=0.0,
                stall_cycles=0.0
            )
        
        # Analyze LSQ occupancy
        lsq_occupancy = self._estimate_lsq_occupancy(memory_ops)
        
        # Calculate stalls due to full LSQ
        stall_cycles = self._calculate_lsq_stalls(memory_ops, lsq_occupancy)
        
        # Calculate memory dependency stalls
        dependency_stalls = self._calculate_dependency_stalls(memory_ops)
        
        total_stalls = stall_cycles + dependency_stalls
        
        # Calculate latency
        latency = self._estimate_memory_latency(memory_ops)
        
        # Calculate throughput impact
        throughput = self._estimate_throughput_impact(memory_ops, total_stalls)
        
        # Utilization
        utilization = min(1.0, lsq_occupancy / self.lsq_size)
        
        return PerformanceMetrics(
            latency=latency,
            throughput=throughput,
            utilization=utilization,
            stall_cycles=total_stalls,
            additional_metrics={
                'avg_lsq_occupancy': lsq_occupancy,
                'load_count': sum(1 for op in memory_ops if op.get('instruction_type') == 'load'),
                'store_count': sum(1 for op in memory_ops if op.get('instruction_type') == 'store'),
                'dependency_stalls': dependency_stalls,
                'capacity_stalls': stall_cycles
            }
        )
    
    def estimate_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate LSQ latency contribution."""
        memory_ops = self._extract_memory_operations(trace_data)
        return self._estimate_memory_latency(memory_ops)
    
    def estimate_throughput(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate throughput impact from LSQ constraints."""
        memory_ops = self._extract_memory_operations(trace_data)
        lsq_occupancy = self._estimate_lsq_occupancy(memory_ops)
        stall_cycles = self._calculate_lsq_stalls(memory_ops, lsq_occupancy)
        dependency_stalls = self._calculate_dependency_stalls(memory_ops)
        return self._estimate_throughput_impact(memory_ops, stall_cycles + dependency_stalls)
    
    def get_stall_cycles(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate stall cycles due to LSQ constraints."""
        memory_ops = self._extract_memory_operations(trace_data)
        lsq_occupancy = self._estimate_lsq_occupancy(memory_ops)
        stall_cycles = self._calculate_lsq_stalls(memory_ops, lsq_occupancy)
        dependency_stalls = self._calculate_dependency_stalls(memory_ops)
        return stall_cycles + dependency_stalls
    
    def get_utilization(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate LSQ utilization."""
        memory_ops = self._extract_memory_operations(trace_data)
        lsq_occupancy = self._estimate_lsq_occupancy(memory_ops)
        return min(1.0, lsq_occupancy / self.lsq_size)
    
    def _extract_memory_operations(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract load and store instructions from trace."""
        return [entry for entry in trace_data 
                if entry.get('instruction_type') in ['load', 'store']]
    
    def _estimate_lsq_occupancy(self, memory_ops: List[Dict[str, Any]]) -> float:
        """
        Estimate average LSQ occupancy.
        
        Based on:
        - Memory operation issue rate
        - Memory operation completion latency
        - Store buffer occupancy
        """
        if not memory_ops:
            return 0.0
        
        # Estimate average memory latency
        avg_latency = self._estimate_avg_memory_latency(memory_ops)
        
        # Issue rate (limited by load/store units)
        issue_rate = min(self.load_store_units, len(memory_ops) / max(1, len(memory_ops)))
        
        # LSQ occupancy = issue_rate * avg_latency
        occupancy = issue_rate * avg_latency
        
        return min(self.lsq_size, occupancy)
    
    def _estimate_avg_memory_latency(self, memory_ops: List[Dict[str, Any]]) -> float:
        """Estimate average memory operation latency."""
        if not memory_ops:
            return 1.0
        
        # Base latency: loads are slower, stores can be buffered
        load_latency = 4.0  # Assuming L1 hit
        store_latency = 1.0  # Store buffer
        
        total_latency = 0.0
        for op in memory_ops:
            if op.get('instruction_type') == 'load':
                total_latency += load_latency
            else:
                total_latency += store_latency
        
        return total_latency / len(memory_ops)
    
    def _calculate_lsq_stalls(self, memory_ops: List[Dict[str, Any]], 
                             lsq_occupancy: float) -> float:
        """Calculate stall cycles when LSQ is full."""
        if lsq_occupancy < self.lsq_size:
            return 0.0
        
        overflow = lsq_occupancy - self.lsq_size
        stall_probability = overflow / self.lsq_size
        
        total_cycles = len(memory_ops)
        stall_cycles = total_cycles * stall_probability
        
        return stall_cycles
    
    def _calculate_dependency_stalls(self, memory_ops: List[Dict[str, Any]]) -> float:
        """
        Calculate stalls due to memory dependencies.
        
        Includes:
        - Load-after-store dependencies
        - Store-after-load dependencies
        - Memory disambiguation delays
        """
        if not memory_ops:
            return 0.0
        
        # Count potential dependencies
        dependency_count = 0
        
        for i, op in enumerate(memory_ops):
            if 'memory_address' not in op:
                continue
            
            addr = op.get('memory_address')
            op_type = op.get('instruction_type')
            
            # Check for dependencies with previous operations
            for j in range(max(0, i - 10), i):  # Look back 10 operations
                prev_op = memory_ops[j]
                if 'memory_address' not in prev_op:
                    continue
                
                prev_addr = prev_op.get('memory_address')
                prev_type = prev_op.get('instruction_type')
                
                # Load after store (potential forwarding)
                if op_type == 'load' and prev_type == 'store':
                    if self._addresses_match(addr, prev_addr):
                        dependency_count += 1
                
                # Store after load (potential ordering issue)
                if op_type == 'store' and prev_type == 'load':
                    if self._addresses_match(addr, prev_addr):
                        dependency_count += 1
        
        # Each dependency may cause a stall
        # Assume 30% of dependencies cause stalls (some can be forwarded)
        stall_rate = 0.3
        avg_stall_per_dep = 2.0  # cycles
        
        return dependency_count * stall_rate * avg_stall_per_dep
    
    def _addresses_match(self, addr1: int, addr2: int, cache_line_size: int = 64) -> bool:
        """Check if two addresses are in the same cache line (potential conflict)."""
        line1 = addr1 // cache_line_size
        line2 = addr2 // cache_line_size
        return line1 == line2
    
    def _estimate_memory_latency(self, memory_ops: List[Dict[str, Any]]) -> float:
        """Estimate total memory operation latency."""
        if not memory_ops:
            return 0.0
        
        avg_latency = self._estimate_avg_memory_latency(memory_ops)
        return len(memory_ops) * avg_latency
    
    def _estimate_throughput_impact(self, memory_ops: List[Dict[str, Any]], 
                                   stall_cycles: float) -> float:
        """Estimate throughput impact from LSQ stalls."""
        if not memory_ops:
            return 1.0
        
        total_cycles = len(memory_ops) + stall_cycles
        return len(memory_ops) / total_cycles if total_cycles > 0 else 1.0


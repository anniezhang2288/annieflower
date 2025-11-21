"""
Memory System Performance Model.
Models the performance impact of memory bandwidth, latency, and contention.
"""

from typing import List, Dict, Any
import numpy as np
from .base_model import BasePerformanceModel, PerformanceMetrics, MicroarchConfig


class MemoryModel(BasePerformanceModel):
    """
    Analytical model for memory system performance.
    
    Models:
    - Memory bandwidth utilization
    - Memory latency
    - Memory contention
    - Prefetching effectiveness
    """
    
    def __init__(self, config: MicroarchConfig):
        super().__init__(config)
        self.memory_latency = config.memory_latency
        self.memory_bandwidth = 25.6  # GB/s (typical DDR4)
        self.cache_line_size = 64  # bytes
    
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Process trace and compute memory system performance metrics."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        
        if not memory_accesses:
            return PerformanceMetrics(
                latency=0.0,
                throughput=1.0,
                utilization=0.0,
                stall_cycles=0.0
            )
        
        # Calculate memory bandwidth usage
        bandwidth_usage = self._calculate_bandwidth_usage(memory_accesses)
        
        # Calculate memory latency
        latency = self._calculate_memory_latency(memory_accesses)
        
        # Calculate contention stalls
        contention_stalls = self._calculate_contention_stalls(memory_accesses, bandwidth_usage)
        
        # Calculate throughput impact
        throughput = self._estimate_throughput_impact(memory_accesses, contention_stalls)
        
        # Utilization (bandwidth utilization)
        utilization = min(1.0, bandwidth_usage / self.memory_bandwidth)
        
        return PerformanceMetrics(
            latency=latency,
            throughput=throughput,
            utilization=utilization,
            stall_cycles=contention_stalls,
            additional_metrics={
                'bandwidth_usage_gbps': bandwidth_usage,
                'bandwidth_utilization': utilization,
                'memory_access_count': len(memory_accesses),
                'avg_memory_latency': latency / len(memory_accesses) if memory_accesses else 0.0
            }
        )
    
    def estimate_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate memory system latency."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        return self._calculate_memory_latency(memory_accesses)
    
    def estimate_throughput(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate throughput impact from memory system."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        bandwidth_usage = self._calculate_bandwidth_usage(memory_accesses)
        contention_stalls = self._calculate_contention_stalls(memory_accesses, bandwidth_usage)
        return self._estimate_throughput_impact(memory_accesses, contention_stalls)
    
    def get_stall_cycles(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate stall cycles due to memory system."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        bandwidth_usage = self._calculate_bandwidth_usage(memory_accesses)
        return self._calculate_contention_stalls(memory_accesses, bandwidth_usage)
    
    def get_utilization(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate memory bandwidth utilization."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        bandwidth_usage = self._calculate_bandwidth_usage(memory_accesses)
        return min(1.0, bandwidth_usage / self.memory_bandwidth)
    
    def _extract_memory_accesses(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract memory accesses that go to main memory (cache misses)."""
        # In a real implementation, this would filter for L3 misses
        # For now, we'll use all memory operations as a proxy
        return [entry for entry in trace_data 
                if entry.get('instruction_type') in ['load', 'store']
                and 'memory_address' in entry]
    
    def _calculate_bandwidth_usage(self, memory_accesses: List[Dict[str, Any]]) -> float:
        """
        Calculate memory bandwidth usage in GB/s.
        
        Assumes:
        - Each access transfers a cache line (64 bytes)
        - Accesses are spread over execution time
        """
        if not memory_accesses:
            return 0.0
        
        # Estimate execution time (simplified)
        # Assume each instruction takes ~1 cycle on average
        execution_time_cycles = len(memory_accesses) * 10  # Rough estimate
        execution_time_seconds = execution_time_cycles / 3.0e9  # Assume 3GHz
        
        # Total data transferred
        bytes_transferred = len(memory_accesses) * self.cache_line_size
        gb_transferred = bytes_transferred / (1024 ** 3)
        
        # Bandwidth usage
        bandwidth_usage = gb_transferred / execution_time_seconds if execution_time_seconds > 0 else 0.0
        
        return bandwidth_usage
    
    def _calculate_memory_latency(self, memory_accesses: List[Dict[str, Any]]) -> float:
        """Calculate total memory access latency."""
        if not memory_accesses:
            return 0.0
        
        # Base memory latency
        base_latency = self.memory_latency
        
        # Add bandwidth-related latency if bandwidth is saturated
        bandwidth_usage = self._calculate_bandwidth_usage(memory_accesses)
        if bandwidth_usage > self.memory_bandwidth * 0.8:
            # Bandwidth saturation adds latency
            saturation_factor = bandwidth_usage / self.memory_bandwidth
            additional_latency = (saturation_factor - 0.8) * 20.0  # Additional cycles
            base_latency += additional_latency
        
        return len(memory_accesses) * base_latency
    
    def _calculate_contention_stalls(self, memory_accesses: List[Dict[str, Any]], 
                                    bandwidth_usage: float) -> float:
        """
        Calculate stall cycles due to memory contention.
        
        Contention occurs when:
        - Memory bandwidth is saturated
        - Multiple memory requests compete for resources
        """
        if bandwidth_usage < self.memory_bandwidth * 0.8:
            return 0.0
        
        # Calculate saturation level
        saturation = min(1.0, bandwidth_usage / self.memory_bandwidth)
        
        # Stalls increase with saturation
        # At 100% saturation, each access may stall
        stall_probability = (saturation - 0.8) / 0.2  # 0 to 1 as saturation goes from 80% to 100%
        stall_probability = max(0.0, min(1.0, stall_probability))
        
        # Average stall per contended access
        avg_stall = 5.0  # cycles
        
        return len(memory_accesses) * stall_probability * avg_stall
    
    def _estimate_throughput_impact(self, memory_accesses: List[Dict[str, Any]], 
                                   contention_stalls: float) -> float:
        """Estimate throughput impact from memory system."""
        if not memory_accesses:
            return 1.0
        
        total_cycles = len(memory_accesses) + contention_stalls
        return len(memory_accesses) / total_cycles if total_cycles > 0 else 1.0


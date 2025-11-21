"""
Cache Hierarchy Performance Model.
Models the performance impact of cache hits, misses, and hierarchy.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from .base_model import BasePerformanceModel, PerformanceMetrics, MicroarchConfig


class CacheModel(BasePerformanceModel):
    """
    Analytical model for cache hierarchy performance.
    
    Models:
    - L1, L2, L3 cache hit/miss rates
    - Cache access latency
    - Memory access latency
    - Cache coherence overhead
    """
    
    def __init__(self, config: MicroarchConfig):
        super().__init__(config)
        self.l1d_config = {
            'size': config.l1d_size,
            'assoc': config.l1d_assoc,
            'latency': config.l1d_latency
        }
        self.l1i_config = {
            'size': config.l1i_size,
            'assoc': config.l1i_assoc,
            'latency': config.l1i_latency
        }
        self.l2_config = {
            'size': config.l2_size,
            'assoc': config.l2_assoc,
            'latency': config.l2_latency
        }
        self.l3_config = {
            'size': config.l3_size,
            'assoc': config.l3_assoc,
            'latency': config.l3_latency
        }
        self.memory_latency = config.memory_latency
    
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Process trace and compute cache performance metrics."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        
        if not memory_accesses:
            return PerformanceMetrics(
                latency=0.0,
                throughput=1.0,
                utilization=0.0,
                stall_cycles=0.0
            )
        
        # Calculate cache hit/miss rates
        l1_hits, l1_misses = self._calculate_l1_stats(memory_accesses)
        l2_hits, l2_misses = self._calculate_l2_stats(memory_accesses, l1_misses)
        l3_hits, l3_misses = self._calculate_l3_stats(memory_accesses, l2_misses)
        
        # Calculate total latency
        latency = self._calculate_total_latency(
            l1_hits, l1_misses, l2_hits, l2_misses, l3_hits, l3_misses
        )
        
        # Calculate stall cycles
        stall_cycles = self._calculate_stall_cycles(
            l1_misses, l2_misses, l3_misses
        )
        
        # Calculate throughput impact
        throughput = self._estimate_throughput_impact(stall_cycles, len(trace_data))
        
        # Utilization
        utilization = len(memory_accesses) / len(trace_data) if trace_data else 0.0
        
        total_accesses = len(memory_accesses)
        l1_hit_rate = l1_hits / total_accesses if total_accesses > 0 else 0.0
        l2_hit_rate = l2_hits / total_accesses if total_accesses > 0 else 0.0
        l3_hit_rate = l3_hits / total_accesses if total_accesses > 0 else 0.0
        memory_access_rate = l3_misses / total_accesses if total_accesses > 0 else 0.0
        
        return PerformanceMetrics(
            latency=latency,
            throughput=throughput,
            utilization=utilization,
            stall_cycles=stall_cycles,
            additional_metrics={
                'l1_hit_rate': l1_hit_rate,
                'l2_hit_rate': l2_hit_rate,
                'l3_hit_rate': l3_hit_rate,
                'memory_access_rate': memory_access_rate,
                'total_memory_accesses': total_accesses,
                'l1_hits': l1_hits,
                'l1_misses': l1_misses,
                'l2_hits': l2_hits,
                'l2_misses': l2_misses,
                'l3_hits': l3_hits,
                'l3_misses': l3_misses
            }
        )
    
    def estimate_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate cache access latency."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        if not memory_accesses:
            return 0.0
        
        l1_hits, l1_misses = self._calculate_l1_stats(memory_accesses)
        l2_hits, l2_misses = self._calculate_l2_stats(memory_accesses, l1_misses)
        l3_hits, l3_misses = self._calculate_l3_stats(memory_accesses, l2_misses)
        
        return self._calculate_total_latency(
            l1_hits, l1_misses, l2_hits, l2_misses, l3_hits, l3_misses
        )
    
    def estimate_throughput(self, trace_data: List[Dict[str, Any]]) -> float:
        """Estimate throughput impact from cache hierarchy."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        if not memory_accesses:
            return 1.0
        
        l1_hits, l1_misses = self._calculate_l1_stats(memory_accesses)
        l2_hits, l2_misses = self._calculate_l2_stats(memory_accesses, l1_misses)
        l3_hits, l3_misses = self._calculate_l3_stats(memory_accesses, l2_misses)
        
        stall_cycles = self._calculate_stall_cycles(l1_misses, l2_misses, l3_misses)
        return self._estimate_throughput_impact(stall_cycles, len(trace_data))
    
    def get_stall_cycles(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate stall cycles due to cache misses."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        if not memory_accesses:
            return 0.0
        
        l1_hits, l1_misses = self._calculate_l1_stats(memory_accesses)
        l2_hits, l2_misses = self._calculate_l2_stats(memory_accesses, l1_misses)
        l3_hits, l3_misses = self._calculate_l3_stats(memory_accesses, l2_misses)
        
        return self._calculate_stall_cycles(l1_misses, l2_misses, l3_misses)
    
    def get_utilization(self, trace_data: List[Dict[str, Any]]) -> float:
        """Calculate cache utilization."""
        memory_accesses = self._extract_memory_accesses(trace_data)
        return len(memory_accesses) / len(trace_data) if trace_data else 0.0
    
    def _extract_memory_accesses(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract memory access instructions from trace."""
        return [entry for entry in trace_data 
                if entry.get('instruction_type') in ['load', 'store'] 
                and 'memory_address' in entry]
    
    def _calculate_l1_stats(self, memory_accesses: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Calculate L1 cache hit/miss statistics.
        Uses analytical model based on cache size, associativity, and access patterns.
        """
        if not memory_accesses:
            return 0, 0
        
        # Extract memory addresses
        addresses = [entry.get('memory_address', 0) for entry in memory_accesses]
        
        # Calculate hit rate using analytical model
        hit_rate = self._estimate_cache_hit_rate(
            addresses, self.l1d_config['size'], self.l1d_config['assoc']
        )
        
        total = len(memory_accesses)
        hits = int(total * hit_rate)
        misses = total - hits
        
        return hits, misses
    
    def _calculate_l2_stats(self, memory_accesses: List[Dict[str, Any]], 
                           l1_misses: int) -> Tuple[int, int]:
        """Calculate L2 cache hit/miss statistics for L1 misses."""
        if l1_misses == 0:
            return 0, 0
        
        # Get addresses that missed L1
        addresses = [entry.get('memory_address', 0) 
                    for entry in memory_accesses[-l1_misses:]]
        
        hit_rate = self._estimate_cache_hit_rate(
            addresses, self.l2_config['size'], self.l2_config['assoc']
        )
        
        hits = int(l1_misses * hit_rate)
        misses = l1_misses - hits
        
        return hits, misses
    
    def _calculate_l3_stats(self, memory_accesses: List[Dict[str, Any]], 
                           l2_misses: int) -> Tuple[int, int]:
        """Calculate L3 cache hit/miss statistics for L2 misses."""
        if l2_misses == 0:
            return 0, 0
        
        # Get addresses that missed L2
        addresses = [entry.get('memory_address', 0) 
                    for entry in memory_accesses[-l2_misses:]]
        
        hit_rate = self._estimate_cache_hit_rate(
            addresses, self.l3_config['size'], self.l3_config['assoc']
        )
        
        hits = int(l2_misses * hit_rate)
        misses = l2_misses - hits
        
        return hits, misses
    
    def _estimate_cache_hit_rate(self, addresses: List[int], 
                                cache_size: int, associativity: int) -> float:
        """
        Estimate cache hit rate using analytical model.
        
        Uses simplified model based on:
        - Cache size and associativity
        - Working set size (unique addresses)
        - Temporal locality
        """
        if not addresses:
            return 1.0
        
        unique_addresses = len(set(addresses))
        cache_blocks = cache_size // 64  # Assume 64-byte cache lines
        
        # If working set fits in cache, expect high hit rate
        if unique_addresses <= cache_blocks:
            return 0.95
        
        # Calculate reuse distance (simplified)
        reuse_factor = unique_addresses / cache_blocks
        
        # Higher associativity improves hit rate
        assoc_factor = min(1.0, associativity / 8.0)
        
        # Base hit rate decreases with larger working sets
        base_hit_rate = 0.85 - (min(0.3, np.log2(reuse_factor) * 0.05))
        hit_rate = base_hit_rate + (assoc_factor * 0.1)
        
        return max(0.5, min(0.98, hit_rate))
    
    def _calculate_total_latency(self, l1_hits: int, l1_misses: int,
                                l2_hits: int, l2_misses: int,
                                l3_hits: int, l3_misses: int) -> float:
        """Calculate total cache access latency."""
        latency = 0.0
        
        # L1 hits
        latency += l1_hits * self.l1d_config['latency']
        
        # L1 misses that hit L2
        latency += l2_hits * (self.l1d_config['latency'] + self.l2_config['latency'])
        
        # L2 misses that hit L3
        latency += l3_hits * (self.l1d_config['latency'] + 
                            self.l2_config['latency'] + 
                            self.l3_config['latency'])
        
        # Memory accesses
        latency += l3_misses * (self.l1d_config['latency'] + 
                              self.l2_config['latency'] + 
                              self.l3_config['latency'] + 
                              self.memory_latency)
        
        return latency
    
    def _calculate_stall_cycles(self, l1_misses: int, l2_misses: int, 
                               l3_misses: int) -> float:
        """
        Calculate stall cycles from cache misses.
        Assumes out-of-order execution can hide some latency.
        """
        # L2 access latency (beyond L1)
        l2_stall = l1_misses * (self.l2_config['latency'] - self.l1d_config['latency'])
        
        # L3 access latency (beyond L2)
        l3_stall = l2_misses * (self.l3_config['latency'] - self.l2_config['latency'])
        
        # Memory access latency (beyond L3)
        memory_stall = l3_misses * (self.memory_latency - self.l3_config['latency'])
        
        # Assume OoO can hide 50% of stall cycles
        total_stall = (l2_stall + l3_stall + memory_stall) * 0.5
        
        return max(0.0, total_stall)
    
    def _estimate_throughput_impact(self, stall_cycles: float, 
                                   total_instructions: int) -> float:
        """Estimate throughput impact from cache stalls."""
        if total_instructions == 0:
            return 1.0
        
        total_cycles = total_instructions + stall_cycles
        return total_instructions / total_cycles if total_cycles > 0 else 1.0


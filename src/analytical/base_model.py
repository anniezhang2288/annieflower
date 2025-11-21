"""
Base class for analytical performance models.
Inspired by Concorde's compositional modeling approach.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics from a model."""
    latency: float
    throughput: float
    utilization: float
    stall_cycles: float
    additional_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass
class MicroarchConfig:
    """Microarchitectural configuration parameters."""
    # Pipeline parameters
    pipeline_width: int = 4  # Instructions per cycle
    pipeline_depth: int = 14  # Pipeline stages
    
    # Branch predictor
    bp_size: int = 4096
    bp_assoc: int = 4
    
    # Cache hierarchy
    l1d_size: int = 32 * 1024  # 32KB
    l1d_assoc: int = 8
    l1d_latency: int = 3
    
    l1i_size: int = 32 * 1024  # 32KB
    l1i_assoc: int = 8
    l1i_latency: int = 3
    
    l2_size: int = 256 * 1024  # 256KB
    l2_assoc: int = 8
    l2_latency: int = 12
    
    l3_size: int = 8 * 1024 * 1024  # 8MB
    l3_assoc: int = 16
    l3_latency: int = 40
    
    # Memory
    memory_latency: int = 100  # cycles
    
    # ROB (Reorder Buffer)
    rob_size: int = 192
    
    # LSQ (Load-Store Queue)
    lsq_size: int = 72
    
    # Execution units
    alu_count: int = 4
    fpu_count: int = 2
    load_store_units: int = 2


class BasePerformanceModel(ABC):
    """
    Base class for all analytical performance models.
    
    Each model should:
    1. Process instruction traces (from Tacit)
    2. Compute performance metrics for its component
    3. Provide latency/throughput estimates
    4. Identify bottlenecks
    """
    
    def __init__(self, config: MicroarchConfig):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def process_trace(self, trace_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """
        Process instruction trace data and compute performance metrics.
        
        Args:
            trace_data: List of instruction trace entries from Tacit
            
        Returns:
            PerformanceMetrics object with computed metrics
        """
        pass
    
    @abstractmethod
    def estimate_latency(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Estimate latency contribution from this component.
        
        Args:
            trace_data: Instruction trace entries
            
        Returns:
            Estimated latency in cycles
        """
        pass
    
    @abstractmethod
    def estimate_throughput(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Estimate throughput impact from this component.
        
        Args:
            trace_data: Instruction trace entries
            
        Returns:
            Estimated throughput (instructions per cycle)
        """
        pass
    
    def get_stall_cycles(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Calculate stall cycles due to this component.
        Override in subclasses for component-specific logic.
        """
        return 0.0
    
    def get_utilization(self, trace_data: List[Dict[str, Any]]) -> float:
        """
        Calculate resource utilization (0.0 to 1.0).
        Override in subclasses for component-specific logic.
        """
        return 0.0
    
    def validate_config(self) -> bool:
        """Validate microarchitectural configuration."""
        return True


"""
Instruction trace processor for Tacit traces.
Processes Tacit instruction traces into a format suitable for performance modeling.
"""

from typing import List, Dict, Any, Optional
import json
import csv
from pathlib import Path


class TacitTraceProcessor:
    """
    Processes instruction traces from Tacit.
    
    Tacit traces typically contain:
    - Instruction addresses (PC)
    - Instruction types (load, store, branch, ALU, etc.)
    - Memory addresses accessed
    - Branch outcomes
    - Dependencies
    - Timestamps/cycles
    """
    
    def __init__(self, trace_file: Optional[str] = None):
        self.trace_file = trace_file
        self.trace_data: List[Dict[str, Any]] = []
        
    def load_trace(self, trace_file: str) -> List[Dict[str, Any]]:
        """
        Load instruction trace from file.
        Supports JSON, CSV, or text formats.
        
        Args:
            trace_file: Path to trace file
            
        Returns:
            List of trace entries
        """
        path = Path(trace_file)
        
        if path.suffix == '.json':
            return self._load_json_trace(trace_file)
        elif path.suffix == '.csv':
            return self._load_csv_trace(trace_file)
        else:
            return self._load_text_trace(trace_file)
    
    def _load_json_trace(self, trace_file: str) -> List[Dict[str, Any]]:
        """Load JSON format trace."""
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Normalize trace format
        if isinstance(data, list):
            return self._normalize_trace_entries(data)
        elif isinstance(data, dict) and 'instructions' in data:
            return self._normalize_trace_entries(data['instructions'])
        else:
            raise ValueError(f"Unexpected JSON trace format in {trace_file}")
    
    def _load_csv_trace(self, trace_file: str) -> List[Dict[str, Any]]:
        """Load CSV format trace."""
        trace_entries = []
        with open(trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace_entries.append(self._normalize_trace_entry(row))
        return trace_entries
    
    def _load_text_trace(self, trace_file: str) -> List[Dict[str, Any]]:
        """
        Load text format trace (common Tacit output format).
        Expected format: PC, instruction_type, [additional fields]
        """
        trace_entries = []
        with open(trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse line (adjust based on actual Tacit format)
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                entry = {
                    'pc': int(parts[0], 16) if parts[0].startswith('0x') else int(parts[0]),
                    'instruction_type': parts[1],
                    'raw': line
                }
                
                # Parse additional fields based on instruction type
                if len(parts) > 2:
                    entry['operands'] = parts[2:]
                
                trace_entries.append(entry)
        
        return self._normalize_trace_entries(trace_entries)
    
    def _normalize_trace_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize trace entries to standard format."""
        normalized = []
        for i, entry in enumerate(entries):
            normalized.append(self._normalize_trace_entry(entry, index=i))
        return normalized
    
    def _normalize_trace_entry(self, entry: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """
        Normalize a single trace entry to standard format.
        
        Standard format includes:
        - pc: Program counter
        - instruction_type: Type of instruction (load, store, branch, alu, etc.)
        - memory_address: For load/store instructions
        - branch_taken: For branch instructions
        - cycle: Cycle number (if available)
        - dependencies: List of dependent instruction indices
        """
        normalized = {
            'index': index,
            'pc': entry.get('pc', entry.get('address', entry.get('PC', 0))),
            'instruction_type': entry.get('instruction_type', 
                                         entry.get('type', 
                                                  entry.get('inst_type', 'unknown'))).lower(),
            'cycle': entry.get('cycle', entry.get('timestamp', index)),
        }
        
        # Memory operations
        if 'memory_address' in entry:
            normalized['memory_address'] = entry['memory_address']
        elif 'mem_addr' in entry:
            normalized['memory_address'] = entry['mem_addr']
        elif 'addr' in entry and normalized['instruction_type'] in ['load', 'store']:
            normalized['memory_address'] = entry['addr']
        
        # Branch information
        if 'branch_taken' in entry:
            normalized['branch_taken'] = entry['branch_taken']
        elif 'taken' in entry:
            normalized['branch_taken'] = entry['taken']
        elif normalized['instruction_type'] in ['branch', 'jump', 'call', 'ret']:
            normalized['branch_taken'] = entry.get('taken', False)
        
        # Dependencies
        if 'dependencies' in entry:
            normalized['dependencies'] = entry['dependencies']
        elif 'deps' in entry:
            normalized['dependencies'] = entry['deps']
        else:
            normalized['dependencies'] = []
        
        # Instruction details
        normalized['opcode'] = entry.get('opcode', entry.get('op', ''))
        normalized['raw'] = entry.get('raw', str(entry))
        
        return normalized
    
    def get_instruction_statistics(self, trace_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics about instruction types in trace."""
        stats = {}
        for entry in trace_data:
            inst_type = entry.get('instruction_type', 'unknown')
            stats[inst_type] = stats.get(inst_type, 0) + 1
        return stats
    
    def filter_by_type(self, trace_data: List[Dict[str, Any]], 
                      instruction_type: str) -> List[Dict[str, Any]]:
        """Filter trace entries by instruction type."""
        return [entry for entry in trace_data 
                if entry.get('instruction_type') == instruction_type]
    
    def get_memory_accesses(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all memory access instructions."""
        return [entry for entry in trace_data 
                if entry.get('instruction_type') in ['load', 'store']]


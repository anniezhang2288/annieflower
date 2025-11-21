"""
Parser for RISC-V instruction trace format.
Handles traces with format: START PC <addr> INST <inst> ... TIMESTAMP <cycle> END
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class RISCVTraceParser:
    """
    Parser for RISC-V instruction traces.
    
    Format:
    START PC <address> INST <instruction> [operands] TIMESTAMP <cycle> END
    BBTIME <cycles> ENDBB
    """
    
    # RISC-V instruction type mapping
    INSTRUCTION_TYPE_MAP = {
        # Load instructions
        'lw': 'load',
        'ld': 'load',
        'lh': 'load',
        'lb': 'load',
        'lhu': 'load',
        'lbu': 'load',
        'lwu': 'load',
        'flw': 'load',
        'fld': 'load',
        
        # Store instructions
        'sw': 'store',
        'sd': 'store',
        'sh': 'store',
        'sb': 'store',
        'fsw': 'store',
        'fsd': 'store',
        
        # Branch instructions
        'beq': 'branch',
        'bne': 'branch',
        'blt': 'branch',
        'bge': 'branch',
        'bltu': 'branch',
        'bgeu': 'branch',
        'jal': 'branch',
        'jalr': 'branch',
        'c.j': 'branch',
        'c.jal': 'branch',
        'c.jr': 'branch',
        'c.jalr': 'branch',
        'c.beqz': 'branch',
        'c.bnez': 'branch',
        
        # ALU instructions
        'add': 'alu',
        'addi': 'alu',
        'sub': 'alu',
        'sll': 'alu',
        'slli': 'alu',
        'srl': 'alu',
        'srli': 'alu',
        'sra': 'alu',
        'srai': 'alu',
        'and': 'alu',
        'andi': 'alu',
        'or': 'alu',
        'ori': 'alu',
        'xor': 'alu',
        'xori': 'alu',
        'slt': 'alu',
        'slti': 'alu',
        'sltu': 'alu',
        'sltiu': 'alu',
        'lui': 'alu',
        'auipc': 'alu',
        'c.add': 'alu',
        'c.addi': 'alu',
        'c.addi16sp': 'alu',
        'c.addi4spn': 'alu',
        'c.andi': 'alu',
        'c.li': 'alu',
        'c.lui': 'alu',
        'c.mv': 'alu',
        'c.slli': 'alu',
        'c.srli': 'alu',
        'c.srai': 'alu',
        'c.sub': 'alu',
        'c.xor': 'alu',
        'c.and': 'alu',
        'c.or': 'alu',
        
        # Memory/System
        'c.sdsp': 'store',
        'c.swsp': 'store',
        'c.ldsp': 'load',
        'c.lwsp': 'load',
        'c.sd': 'store',
        'c.sw': 'store',
        'c.ld': 'load',
        'c.lw': 'load',
        
        # Floating point
        'fadd': 'fpu',
        'fsub': 'fpu',
        'fmul': 'fpu',
        'fdiv': 'fpu',
        'fsqrt': 'fpu',
        'fmv': 'fpu',
        'fmv.w.x': 'fpu',
        'fmv.x.w': 'fpu',
        
        # System
        'csrrw': 'system',
        'csrrs': 'system',
        'csrrc': 'system',
        'csrrwi': 'system',
        'csrrsi': 'system',
        'csrrci': 'system',
        'ecall': 'system',
        'ebreak': 'system',
        'mret': 'system',
        'wfi': 'system',
    }
    
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.trace_data: List[Dict[str, Any]] = []
        
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse the RISC-V trace file.
        
        Returns:
            List of normalized trace entries
        """
        trace_entries = []
        current_cycle = 0
        instruction_index = 0
        
        with open(self.trace_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse instruction line
                if line.startswith('START PC'):
                    entry = self._parse_instruction_line(line, instruction_index, current_cycle)
                    if entry:
                        trace_entries.append(entry)
                        instruction_index += 1
                        # Update cycle from timestamp if available
                        if 'cycle' in entry:
                            current_cycle = entry['cycle']
                        else:
                            current_cycle += 1
                
                # Parse basic block timing (optional, for analysis)
                elif line.startswith('BBTIME'):
                    # Could use this for basic block analysis
                    pass
        
        return trace_entries
    
    def _parse_instruction_line(self, line: str, index: int, default_cycle: int) -> Optional[Dict[str, Any]]:
        """
        Parse a single instruction line.
        
        Format: START PC <addr> INST <inst> [operands] TIMESTAMP <cycle> END
        """
        # Extract PC
        pc_match = re.search(r'PC\s+([0-9a-fA-F]+)', line)
        if not pc_match:
            return None
        
        pc = int(pc_match.group(1), 16)
        
        # Extract instruction
        inst_match = re.search(r'INST\s+(\S+)', line)
        if not inst_match:
            return None
        
        inst_name = inst_match.group(1)
        
        # Map to instruction type
        inst_type = self.INSTRUCTION_TYPE_MAP.get(inst_name, 'unknown')
        
        # Extract timestamp
        timestamp_match = re.search(r'TIMESTAMP\s+(\d+)', line)
        cycle = int(timestamp_match.group(1)) if timestamp_match else default_cycle
        
        # Extract register operands
        rd_match = re.search(r'RD\s+(\S+)', line)
        rs1_match = re.search(r'RS1\s+(\S+)', line)
        rs2_match = re.search(r'RS2\s+(\S+)', line)
        
        # Extract immediate
        imm_match = re.search(r'IMM\s+(-?\d+)', line)
        imm = int(imm_match.group(1)) if imm_match else None
        
        # Extract memory address for load/store
        memory_address = None
        if inst_type in ['load', 'store']:
            # For load/store, memory address = RS1 + IMM
            if rs1_match and imm_match:
                # This is simplified - in reality we'd need register values
                # For now, we'll use a placeholder
                memory_address = imm  # Simplified
        
        # Determine branch outcome (simplified - would need actual execution)
        branch_taken = None
        if inst_type == 'branch':
            # For unconditional branches (jal, jalr), always taken
            if inst_name in ['jal', 'jalr', 'c.j', 'c.jal', 'c.jr', 'c.jalr']:
                branch_taken = True
            # For conditional branches, we can't determine without execution state
            # This would need to be enhanced with actual branch outcomes
        
        entry = {
            'index': index,
            'pc': pc,
            'instruction_type': inst_type,
            'instruction_name': inst_name,
            'cycle': cycle,
            'raw': line
        }
        
        if rd_match:
            entry['rd'] = rd_match.group(1)
        if rs1_match:
            entry['rs1'] = rs1_match.group(1)
        if rs2_match:
            entry['rs2'] = rs2_match.group(1)
        if imm is not None:
            entry['imm'] = imm
        if memory_address is not None:
            entry['memory_address'] = memory_address
        if branch_taken is not None:
            entry['branch_taken'] = branch_taken
        
        return entry
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the parsed trace."""
        if not self.trace_data:
            self.trace_data = self.parse()
        
        stats = {
            'total_instructions': len(self.trace_data),
            'instruction_types': {},
            'unique_pcs': len(set(e['pc'] for e in self.trace_data)),
            'cycles': 0,
            'branches': 0,
            'loads': 0,
            'stores': 0,
        }
        
        if self.trace_data:
            stats['cycles'] = max(e.get('cycle', 0) for e in self.trace_data)
            
            for entry in self.trace_data:
                inst_type = entry['instruction_type']
                stats['instruction_types'][inst_type] = stats['instruction_types'].get(inst_type, 0) + 1
                
                if inst_type == 'branch':
                    stats['branches'] += 1
                elif inst_type == 'load':
                    stats['loads'] += 1
                elif inst_type == 'store':
                    stats['stores'] += 1
        
        return stats


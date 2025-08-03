use std::fmt;
use serde::{Deserialize, Serialize};
use crate::{BasicBlock, IndexRegScale, Instruction, Operand, RegisterId};

pub type SymRegisterId = u16;

// FIXME: maybe make field generic?
//  currently this struct is exact same and non-symbolized
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolizedBasicBlock {
    pub instructions: Box<[SymbolizedInstruction]>,
}

impl fmt::Display for SymbolizedBasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined = self.instructions.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(";\n");
        write!(f, "{joined};")
    }
}

// FIXME: maybe make field generic?
//  currently this struct is exact same and non-symbolized
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolizedInstruction {
    pub opcode: u16,
    pub src: Box<[SymbolizedOperand]>,
    pub dst: Box<[SymbolizedOperand]>,
}

impl fmt::Display for SymbolizedInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let src_str = self.src.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        let dst_str = self.dst.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        if self.dst.is_empty() {
            write!(f, "{}, {}", self.opcode, src_str)
        } else {
            write!(f, "{}, {} -> {}", self.opcode, src_str, dst_str)
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum SymbolizedOperand {
    Immediate,
    Register(SymRegisterId),
    MemoryReference {
        base: SymRegisterId,
        index: Option<(SymRegisterId, IndexRegScale)>,
    },
    Address,
}

impl fmt::Display for SymbolizedOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolizedOperand::Immediate => write!(f, "IMM"),
            SymbolizedOperand::Register(id) => write!(f, "REG{id}"),
            SymbolizedOperand::MemoryReference { base, index: Some((index, scale)) } => {
                write!(f, "[REG{base} + REG{index}*{} + IMM]", *scale as u8)
            }
            SymbolizedOperand::MemoryReference { base, .. } => write!(f, "[REG{base} + IMM]"),
            SymbolizedOperand::Address => write!(f, "ADDR"),
        }
    }
}

#[derive(Clone, Default, Debug, Eq, PartialEq, Hash)]
pub struct RegisterMapping(pub Vec<RegisterId>);

impl RegisterMapping {
    pub fn map(&mut self, reg: RegisterId) -> SymRegisterId {
        if let Some(sym_id) = self.0.iter().position(|id| *id == reg) {
            sym_id as _
        } else {
            self.0.push(reg);
            (self.0.len() - 1) as _
        }
    }
}

impl Operand {
    pub fn symbolize(&self, reg_map: &mut RegisterMapping) -> SymbolizedOperand {
        use Operand as Opnd;
        use SymbolizedOperand as SymOpnd;

        match self {
            Opnd::ImmediateInt(_) | Operand::ImmediateFloat(_) => SymOpnd::Immediate,
            Opnd::Register(id) => SymOpnd::Register(reg_map.map(*id)),
            Opnd::MemoryReference { base, index, .. } => {
                SymbolizedOperand::MemoryReference {
                    base: reg_map.map(*base),
                    index: index.map(|(id, scale)| (reg_map.map(id), scale))
                }
            }
            Opnd::Address(_) => SymOpnd::Address,
        }
    }
}

impl Instruction {
    pub fn symbolize(&self, reg_map: &mut RegisterMapping) -> SymbolizedInstruction {
        SymbolizedInstruction {
            opcode: self.opcode,
            src: self.src.iter().map(|op| op.symbolize(reg_map)).collect(),
            dst: self.dst.iter().map(|op| op.symbolize(reg_map)).collect(),
        }
    }
}

impl BasicBlock {
    pub fn symbolize(&self) -> SymbolizedBasicBlock {
        let mut reg_map = RegisterMapping::default();

        SymbolizedBasicBlock {
            instructions: self.instructions.iter().map(|instr| instr.symbolize(&mut reg_map)).collect(),
        }
    }
}
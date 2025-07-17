use serde::{Deserialize, Serialize};
use crate::IndexRegScale;

pub type SymRegisterId = u16;

// FIXME: maybe make field generic?
//  currently this struct is exact same and non-symbolized
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolizedBasicBlock {
    pub instructions: Box<[SymbolizedInstruction]>,
}

// FIXME: maybe make field generic?
//  currently this struct is exact same and non-symbolized
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymbolizedInstruction {
    pub opcode: u16,
    pub src: Box<[SymbolizedOperand]>,
    pub dst: Box<[SymbolizedOperand]>,
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
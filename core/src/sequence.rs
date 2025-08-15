use std::fmt;
use serde::{Deserialize, Serialize};
use crate::{BasicBlock, BasicBlockLocation as BlockLoc, SymbolizedBasicBlock};

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BasicBlockSequence(pub Box<[(BlockLoc, BasicBlock)]>);

impl BasicBlockSequence {
    pub fn blocks(&self) -> impl Iterator<Item=&BasicBlock> {
        self.0
            .iter()
            .map(|(_, block)| block)
    }

    pub fn symbolized_blocks(&self) -> impl Iterator<Item=SymbolizedBasicBlock> {
        self.blocks().map(BasicBlock::symbolize)
    }
}

impl fmt::Display for BasicBlockSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined = self.0.iter()
            .map(|(loc, block)| format!("{loc}:\n{block}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        write!(f, "{joined}")
    }
}
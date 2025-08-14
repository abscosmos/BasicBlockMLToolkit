use crate::{BasicBlock, BasicBlockLocation as BlockLoc, SymbolizedBasicBlock};

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
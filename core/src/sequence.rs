use crate::{BasicBlock, BasicBlockLocation as BlockLoc};

pub struct BasicBlockSequence(pub Box<[(BlockLoc, BasicBlock)]>);

impl BasicBlockSequence {
    pub fn blocks(&self) -> impl Iterator<Item=&BasicBlock> {
        self.0
            .iter()
            .map(|(_, block)| block)
    }
}
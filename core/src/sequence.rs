use crate::{BasicBlock, BasicBlockLocation as BlockLoc};

pub struct BasicBlockSequence {
    pub blocks: Box<[(BlockLoc, BasicBlock)]>,
}
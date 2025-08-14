use ordermap::OrderMap;
use crate::{BasicBlock, BasicBlockLocation as BlockLoc};

pub struct BasicBlockSequence {
    pub blocks: OrderMap<BlockLoc, BasicBlock>,
}
use crate::{BasicBlock, BasicBlockLocation as BlockLoc};

pub struct BasicBlockSequence(pub Box<[(BlockLoc, BasicBlock)]>);
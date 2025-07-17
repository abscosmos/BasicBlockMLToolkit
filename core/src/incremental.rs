use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use crate::BasicBlock;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Incremental(pub HashSet<BasicBlock>);
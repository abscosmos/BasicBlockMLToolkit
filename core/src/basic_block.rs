use std::fmt;
use std::num::NonZeroUsize;
use serde::{Deserialize, Serialize};
use crate::Instruction;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Application {
    pub name: Box<str>,
    pub address: NonZeroUsize,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BasicBlockLocation {
    pub application: Application,
    pub relative_addr: usize,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct BasicBlock {
    pub instructions: Box<[Instruction]>,
}

// FIXME: duplicated code
impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined = self.instructions.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(";\n");
        write!(f, "{joined};")
    }
}
use serde::{Deserialize, Serialize};
use crate::Instruction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Application {
    pub name: String,
    pub address: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlockLocation {
    pub application: Application,
    pub relative_addr: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub instructions: Box<Instruction>,
}
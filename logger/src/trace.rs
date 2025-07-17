use hashbrown::HashMap;
use logger_core::{Application, BasicBlock, BasicBlockLocation as BlockLoc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TraceData {
    pub targeted: Application,
    pub filter: bool,
    pub blocks: HashMap<BlockLoc, BasicBlock>,
}
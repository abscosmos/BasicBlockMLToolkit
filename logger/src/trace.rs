use hashbrown::{HashMap, HashSet};
use logger_core::{Application, BasicBlock, BasicBlockLocation as BlockLoc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    pub targeted: Application,
    pub filter: bool,
    pub blocks: HashMap<BlockLoc, BasicBlock>,
}

impl TraceData {
    pub fn summary(&self) -> TraceSummary {
        fn unique_ct<'a>(iter: impl IntoIterator<Item=&'a BasicBlock>) -> usize {
            iter.into_iter().collect::<HashSet<_>>().len()
        }

        let targeted = if !self.filter {
            let iter = self.blocks
                .iter()
                .filter_map(|(k, v)|
                    (k.application == self.targeted).then_some(v)
                );


            Some(
                BlockCountStats {
                    num_blocks: iter.clone().count(),
                    num_unique_blocks: unique_ct(iter),
                }
            )
        } else {
            None
        };

        TraceSummary {
            application: self.targeted.clone(),
            counts: BlockCountStats {
                num_blocks: self.blocks.len(),
                num_unique_blocks: unique_ct(self.blocks.values()),
            },
            targeted,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    application: Application,
    counts: BlockCountStats,
    targeted: Option<BlockCountStats>,
}

#[derive(Debug, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct BlockCountStats {
    num_blocks: usize,
    num_unique_blocks: usize,
}
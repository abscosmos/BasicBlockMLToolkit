use std::hash::Hash;
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::{Application, BasicBlock, BasicBlockLocation as BlockLoc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    pub targeted: Application,
    pub filter: bool,
    pub blocks: HashMap<BlockLoc, BasicBlock>,
}

impl TraceData {
    pub fn summary(&self) -> TraceSummary {
        fn unique_ct<T: Hash + Eq>(iter: impl IntoIterator<Item=T>) -> usize {
            iter.into_iter().collect::<HashSet<_>>().len()
        }

        let (targeted, unique_apps) = if !self.filter {
            let iter = self.blocks
                .iter()
                .filter_map(|(k, v)|
                    (k.application == self.targeted).then_some(v)
                );

            let apps = self.blocks
                .keys()
                .map(|k| &k.application);

            let targeted = BlockCountStats {
                num_blocks: iter.clone().count(),
                num_unique_blocks: unique_ct(iter.clone()),
                num_unique_symbolized: unique_ct(iter.map(BasicBlock::symbolize)),
            };

            (Some(targeted), Some(unique_ct(apps)))
        } else {
            (None, None)
        };

        TraceSummary {
            application: self.targeted.clone(),
            counts: BlockCountStats {
                num_blocks: self.blocks.len(),
                num_unique_blocks: unique_ct(self.blocks.values()),
                num_unique_symbolized: unique_ct(self.blocks.values().map(BasicBlock::symbolize)),
            },
            targeted,
            unique_apps,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    application: Application,
    counts: BlockCountStats,
    targeted: Option<BlockCountStats>,
    unique_apps: Option<usize>,
}

#[derive(Debug, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct BlockCountStats {
    num_blocks: usize,
    num_unique_blocks: usize,
    num_unique_symbolized: usize,
}
use hashbrown::HashMap;
use pyo3::{pyclass, pymethods};
use crate::{SymbolizedBasicBlock, TraceData};

#[pyclass]
#[derive(Default, Debug, Clone)]
pub struct BasicBlockTokenizer {
    block_to_token: HashMap<bb_core::SymbolizedBasicBlock, usize>,
    token_to_block: Vec<bb_core::SymbolizedBasicBlock>,
}

#[pymethods]
impl BasicBlockTokenizer {
    #[classattr]
    pub const PADDING_TOKEN: usize = 0;
    #[classattr]
    pub const UNKNOWN_TOKEN: usize = 1;

    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_block(&mut self, block: SymbolizedBasicBlock) -> usize {
        let block = block.0;

        if let Some(token) = self.block_to_token.get(&block) {
            *token
        } else {
            let token = self.next_token();

            self.block_to_token.insert(block.clone(), token);
            self.token_to_block.push(block);

            token
        }
    }

    pub fn get_token(&self, block: SymbolizedBasicBlock) -> usize {
        self.block_to_token.get(&block.0)
            .copied()
            .unwrap_or(Self::UNKNOWN_TOKEN)
    }

    pub fn get_block(&self, token: usize) -> Option<SymbolizedBasicBlock> {
        if token == Self::PADDING_TOKEN || token == Self::UNKNOWN_TOKEN {
            None
        } else {
            self.token_to_block.get(token - 2)
                .cloned()
                .map(SymbolizedBasicBlock)
        }
    }

    pub fn process_trace(&mut self, trace: &TraceData) -> Vec<usize> {
        let mut token_seq = Vec::with_capacity(trace.order.len());

        for block_loc in &trace.order {
            let symbolized = trace.blocks.get(block_loc)
                .expect("all block locs in order should exist in mapping")
                .symbolize();

            token_seq.push(self.add_block(symbolized));
        }

        token_seq
    }

    pub fn __len__(&self) -> usize {
        self.next_token()
    }
}

impl BasicBlockTokenizer {
    const SPECIAL_TOKEN_COUNT: usize = 2;

    fn from_internal_vec(token_to_block: Vec<bb_core::SymbolizedBasicBlock>) -> Self {
        let mut block_to_token = HashMap::with_capacity(token_to_block.len());

        for (i, block) in token_to_block.iter().enumerate() {
            block_to_token.insert(block.clone(), i + Self::SPECIAL_TOKEN_COUNT);
        }

        Self {
            block_to_token,
            token_to_block,
        }
    }

    fn next_token(&self) -> usize {
        self.token_to_block.len() + Self::SPECIAL_TOKEN_COUNT
    }
}
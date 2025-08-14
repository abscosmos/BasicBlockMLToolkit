mod tokenizer;

use pyo3::Bound;
use std::{fmt, fs};
use hashbrown::HashMap;
use std::path::PathBuf;
use pyo3::{pyclass, pymethods, pymodule, PyResult};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyModule, PyModuleMethods};

#[pyclass(eq, hash, frozen, str)]
#[repr(transparent)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BasicBlock(bb_core::BasicBlock);

#[pymethods]
impl BasicBlock {
    pub fn symbolize(&self) -> SymbolizedBasicBlock {
        SymbolizedBasicBlock(self.0.symbolize())
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[pyclass(eq, hash, frozen, str)]
#[repr(transparent)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct SymbolizedBasicBlock(bb_core::SymbolizedBasicBlock);

impl fmt::Display for SymbolizedBasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[pyclass(eq, hash, frozen)]
#[repr(transparent)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BasicBlockLocation(bb_core::BasicBlockLocation);

#[pymethods]
impl BasicBlockLocation {
    #[getter]
    fn relative_addr(&self) -> usize {
        self.0.relative_addr
    }

    #[getter]
    fn application(&self) -> Application {
        Application(self.0.application.clone())
    }
}

#[pyclass(eq, hash, frozen)]
#[repr(transparent)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Application(bb_core::Application);

#[pymethods]
impl Application {
    #[getter]
    fn address(&self) -> usize {
        self.0.address.get()
    }

    #[getter]
    fn name(&self) -> String {
        self.0.name.to_string()
    }
}

#[pyclass(frozen, get_all)]
#[derive(Debug, Clone)]
pub struct TraceData {
    pub targeted: Application,
    pub filter: bool,
    pub blocks: HashMap<BasicBlockLocation, BasicBlock>,
    pub order: Vec<BasicBlockLocation>,
}

#[pymethods]
impl TraceData {
    #[staticmethod]
    pub fn from_binary_file(path: PathBuf) -> PyResult<Self> {
        let file = fs::read(path)?;

        let trace = postcard::from_bytes::<bb_core::TraceData>(&file)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let blocks: HashMap<BasicBlockLocation, BasicBlock> = trace.blocks
            .into_iter()
            .map(|(loc, block)| (BasicBlockLocation(loc), BasicBlock(block)))
            .collect();

        let order: Vec<BasicBlockLocation> = trace.order
            .into_iter()
            .map(BasicBlockLocation)
            .collect();

        Ok(TraceData {
            targeted: Application(trace.targeted),
            filter: trace.filter,
            blocks,
            order,
        })
    }

    pub fn sequence(&self) -> BasicBlockSequence {
        let mut blocks = Vec::with_capacity(self.order.len());

        for loc in &self.order {
            let block = self.blocks.get(loc).expect("corresponding block should exist");

            blocks.push((loc.clone(), block.clone()));
        }

        BasicBlockSequence(blocks)
    }
}

// TODO: remove code duplication for related methods?
#[pyclass(frozen, eq, hash)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BasicBlockSequence(pub Vec<(BasicBlockLocation, BasicBlock)>);

#[pymethods]
impl BasicBlockSequence {
    pub fn blocks(&self) -> Vec<BasicBlock> {
        self.0.iter()
            .map(|(_, block)| block)
            .cloned()
            .collect()
    }

    pub fn symbolized_blocks(&self) -> Vec<SymbolizedBasicBlock> {
        self.blocks()
            .iter()
            .map(BasicBlock::symbolize)
            .collect()
    }
}

#[pymodule]
fn bb_toolkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BasicBlock>()?;
    m.add_class::<SymbolizedBasicBlock>()?;
    m.add_class::<BasicBlockLocation>()?;
    m.add_class::<Application>()?;
    m.add_class::<TraceData>()?;

    m.add_class::<tokenizer::BasicBlockTokenizer>()?;

    Ok(())
}
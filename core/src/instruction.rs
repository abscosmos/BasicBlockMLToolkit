use serde::{Serialize, Deserialize};

pub type RegisterId = u16;

#[derive(Debug, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: u16,
    pub src: Box<[Operand]>,
    pub dst: Box<[Operand]>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Operand {
    ImmediateInt(i64),
    ImmediateFloat(f32),
    Register(RegisterId),
    MemoryReference {
        base: RegisterId,
        index: Option<(RegisterId, IndexRegScale)>,
        displacement: i32,
    },
    Address(usize),
}

#[repr(u8)]
#[derive(Debug, Serialize, Deserialize)]
pub enum IndexRegScale {
    // FIXME: this is only here for testing, remove it
    None = 0,
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
}
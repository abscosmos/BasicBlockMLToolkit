use serde::{Serialize, Deserialize};

pub type RegisterId = u16;

#[derive(Debug, Serialize, Deserialize)]
pub struct Instruction {
    opcode: u16,
    src: Box<[Operand]>,
    dst: Box<[Operand]>,
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
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
}
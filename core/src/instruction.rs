use serde::{Serialize, Deserialize};

pub type RegisterId = u16;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: u16,
    pub src: Box<[Operand]>,
    pub dst: Box<[Operand]>,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum IndexRegScale {
    // FIXME: this is only here for testing, remove it
    None = 0,
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
}

impl IndexRegScale {
    pub fn from_u8(scale: u8) -> Option<Self> {
        let ret = match scale {
            0 => Self::None,
            1 => Self::One,
            2 => Self::Two,
            4 => Self::Four,
            8 => Self::Eight,
            _ => return None,
        };

        debug_assert_eq!(
            ret as u8, scale,
            "repr of IndexRegScale should be same as scale",
        );

        Some(ret)
    }
}
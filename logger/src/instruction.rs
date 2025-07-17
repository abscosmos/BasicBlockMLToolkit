use std::ffi::{c_int, c_uint};
use dynamorio_sys::{instr_get_dst, instr_get_opcode, instr_get_src, instr_num_dsts, instr_num_srcs, instr_t, opnd_get_addr, opnd_get_base, opnd_get_disp, opnd_get_immed_float, opnd_get_immed_int, opnd_get_immed_int64, opnd_get_index, opnd_get_pc, opnd_get_reg, opnd_get_scale, opnd_is_immed, opnd_is_immed_float, opnd_is_immed_int, opnd_is_immed_int64, opnd_is_memory_reference, opnd_is_pc, opnd_is_reg, opnd_is_rel_addr, opnd_t};
use logger_core::{IndexRegScale, Instruction, Operand};

pub fn make_instruction(instr: &mut instr_t) -> Instruction {
    let opcode = unsafe { instr_get_opcode(instr) } as _;

    unsafe fn collect_operands(
        ct: unsafe extern "C" fn(*mut instr_t) -> c_int,
        get: unsafe extern "C" fn(*mut instr_t, c_uint) -> opnd_t,
        instr: &mut instr_t,
    ) -> Box<[Operand]> {
        unsafe {
            (0..ct(instr))
                .map(|i| make_operand(get(instr, i as _)))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        }
    }

    Instruction {
        opcode,
        src: unsafe { collect_operands(instr_num_srcs, instr_get_src, instr) },
        dst: unsafe { collect_operands(instr_num_dsts, instr_get_dst, instr) },
    }
}

fn make_operand(opnd: opnd_t) -> Operand {
    unsafe {
        if opnd_is_reg(opnd) != 0 {
            Operand::Register(opnd_get_reg(opnd))
        } else if opnd_is_immed(opnd) != 0 {
            if opnd_is_immed_int(opnd) != 0  {
                Operand::ImmediateInt(opnd_get_immed_int(opnd))
            } else if opnd_is_immed_int64(opnd) != 0 {
                Operand::ImmediateInt(opnd_get_immed_int64(opnd))
            } else if opnd_is_immed_float(opnd) != 0 {
                Operand::ImmediateFloat(opnd_get_immed_float(opnd).into())
            } else {
                panic!("Unknown operand type! {opnd:?}");
            }
        } else if opnd_is_pc(opnd) != 0 {
            Operand::Address(opnd_get_pc(opnd).addr())
        } else if opnd_is_rel_addr(opnd) != 0 {
            Operand::Address(opnd_get_addr(opnd).addr())
        } else if opnd_is_memory_reference(opnd) != 0 {
            Operand::MemoryReference {
                base: opnd_get_base(opnd),
                index: Some((
                    opnd_get_index(opnd),
                    IndexRegScale::from_u8(opnd_get_scale(opnd) as _)
                        .expect("should be valid scale")
                )),
                displacement: opnd_get_disp(opnd),
            }
        } else {
            panic!("unknown operand type! {opnd:?}")
        }
    }
}
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;
use dynamorio_sys::{bool_, byte as dr_byte, dr_emit_flags_t, dr_free_module_data, dr_lookup_module, dr_memory_is_in_client, instr_disassemble_to_buffer, instr_get_app_pc, instr_get_next_app, instr_t, instrlist_first_app, instrlist_t};
use logger_core::{BasicBlock, BasicBlockLocation as BlockLoc};
use crate::instruction::make_instruction;
use crate::{module_to_application, LOGGER};

pub unsafe extern "C" fn basic_block(
    dr_ctx: *mut c_void,
    _tag: *mut c_void,
    basic_block: *mut instrlist_t,
    _for_trace: bool_,
    _translating: bool_,
) -> dr_emit_flags_t {
    let mut logger_lock = LOGGER.lock();
    let logger = logger_lock.as_mut().expect("logger should be initialized");

    let first_instr = unsafe { instrlist_first_app(basic_block) };

    // address of first instruction
    let start_pc = unsafe { instr_get_app_pc(first_instr) };

    if unsafe { dr_memory_is_in_client(start_pc) } != 0 {
        return dr_emit_flags_t::DR_EMIT_DEFAULT;
    }

    // get the module (executable / library) of the basic block
    let Some(module) = NonNull::new(
        unsafe { dr_lookup_module(start_pc) }
    ) else {
        return dr_emit_flags_t::DR_EMIT_DEFAULT;
    };

    let application = module_to_application(unsafe { module.as_ref() });

    if logger.trace.filter && logger.trace.targeted.address != application.address {
        return dr_emit_flags_t::DR_EMIT_DEFAULT;
    }

    assert!(
        start_pc.addr() > application.address.get(),
        "first instruction should be after the start of the module"
    );

    assert_eq!(
        size_of::<dr_byte>(), 1,
        "relative address arithmetic assumes byte is one byte"
    );


    let block_loc = BlockLoc {
        relative_addr: start_pc.addr() - application.address.get(),
        application,
    };

    if logger.trace.blocks.contains_key(&block_loc) {
        return dr_emit_flags_t::DR_EMIT_DEFAULT;
    }

    let mut instructions = Vec::new();

    // log every instruction
    let mut instr = first_instr;

    while let Some(instr_ref) = unsafe { instr.as_mut() } {
        let _debug_str = debug_instr_str(dr_ctx, instr_ref);

        instructions.push(make_instruction(instr_ref));

        instr = unsafe { instr_get_next_app(instr) };
    }

    // free module data
    unsafe { dr_free_module_data(module.as_ptr()) };

    logger.trace.blocks.insert(
        block_loc,
        BasicBlock {
            instructions: instructions.into_boxed_slice()
        }
    );

    dr_emit_flags_t::DR_EMIT_DEFAULT
}

fn debug_instr_str(dr_ctx: *mut c_void, instr: &mut instr_t) -> String {
    let mut buf = vec![0; 256];

    unsafe {
        instr_disassemble_to_buffer(
            dr_ctx,
            instr,
            buf.as_mut_ptr(),
            buf.len()
        );
    }

    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into()
}
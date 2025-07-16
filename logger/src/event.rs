use std::io::Write as _;
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;
use dynamorio_sys::{bool_, dr_emit_flags_t, dr_free_module_data, dr_lookup_module, dr_module_preferred_name, instr_disassemble_to_buffer, instr_get_app_pc, instr_get_next_app, instr_t, instrlist_first_app, instrlist_t};
use crate::instruction::make_instruction;
use crate::LOGGER;

pub unsafe extern "C" fn basic_block(
    dr_ctx: *mut c_void,
    _tag: *mut c_void,
    basic_block: *mut instrlist_t,
    _for_trace: bool_,
    _translating: bool_,
) -> dr_emit_flags_t {
    let first_instr = unsafe { instrlist_first_app(basic_block) };

    // address of first instruction
    let start_pc = unsafe { instr_get_app_pc(first_instr) };

    // get the module (executable / library) of the basic block
    let Some(module) = NonNull::new(
        unsafe { dr_lookup_module(start_pc) }
    ) else {
        return dr_emit_flags_t::DR_EMIT_DEFAULT;
    };

    let module_start = unsafe { (*module.as_ptr()).__bindgen_anon_1.start };

    assert!(
        start_pc > module_start,
        "first instruction should be after the start of the module"
    );

    // relative address from start of module
    let rel_addr = unsafe { start_pc.offset_from(module_start) } as usize;

    let module_name = match unsafe { dr_module_preferred_name(module.as_ptr()) } {
        ptr if ptr.is_null() => "unknown".into(),
        ptr => unsafe { CStr::from_ptr(ptr) }.to_string_lossy(),
    };

    let mut logger_lock = LOGGER.lock();

    let file = &mut logger_lock
        .as_mut()
        .expect("logger should be initialized")
        .file;

    writeln!(file, "<{module_name}> + {rel_addr:x}").expect("failed to write to log file");

    // log every instruction
    let mut instr = first_instr;

    while !instr.is_null() {
        // isn't null, dynamorio responsibility to ensure valid & well aligned
        let instr_ref = unsafe { &mut *instr };

        let debug_str = debug_instr_str(dr_ctx, instr_ref);

        let instruction = make_instruction(instr_ref);

        writeln!(file, "{debug_str}\n{instruction:?}\n").expect("shouldn't fail to write to file");

        instr = unsafe { instr_get_next_app(instr) };
    }

    // free module data
    unsafe { dr_free_module_data(module.as_ptr()) };

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
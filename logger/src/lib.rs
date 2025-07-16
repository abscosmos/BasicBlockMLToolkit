use std::borrow::Cow;
use std::io::Write as _;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::File;
use std::ptr::NonNull;
use std::slice;
use dynamorio_sys::{bool_, client_id_t, dr_emit_flags_t, dr_lookup_module, dr_module_preferred_name, dr_register_bb_event, dr_register_exit_event, dr_set_client_name, instr_get_app_pc, instrlist_first_app, instrlist_t};
use parking_lot::Mutex;

#[unsafe(no_mangle)]
pub static _USES_DR_VERSION_: c_int = dynamorio_sys::_USES_DR_VERSION_;

struct Logger {
    file: File,
}

static LOGGER: Mutex<Option<Logger>> = Mutex::new(None);

/// # Safety
/// - `argc` and `argv` must be from main call
/// - `argv` should point to `argc` number of valid null terminated c strings
/// - the array pointed to by `argv` & the c strings should be
/// valid for the entire duration of the program (`'static`)
unsafe fn collect_args<'a>(argc: c_int, argv: *const *const c_char) -> impl Iterator<Item=Cow<'static, str>> {
    let args = unsafe { slice::from_raw_parts(argv, argc as _) };

    args.into_iter()
        .map(|&ptr|
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy()
        )
}

#[unsafe(no_mangle)]
pub extern "C" fn dr_client_main(
    _id: client_id_t,
    _argc: c_int,
    _argv: *const *const c_char,
) {
    unsafe {
        dr_set_client_name(c"bblogger".as_ptr(), c"https://github.com/abscosmos/basic-block-trace".as_ptr());

        dr_register_exit_event(Some(exit_event));

        dr_register_bb_event(Some(basic_block_event));
    }

    let logger = Logger {
        file: File::create("bb_trace.log").expect("should be able to create file"),
    };

    *LOGGER.lock() = Some(logger);
}

pub extern "C" fn exit_event() {
    // ensure Drop::drop runs for Logger
    let _ = LOGGER.lock().take();
}

pub unsafe extern "C" fn basic_block_event(
    _dr_ctx: *mut c_void,
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

    dr_emit_flags_t::DR_EMIT_DEFAULT
}
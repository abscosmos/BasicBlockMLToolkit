use std::ffi::{c_char, c_int};
use std::fs::File;
use dynamorio_sys::{client_id_t, dr_register_exit_event, dr_set_client_name};
use parking_lot::Mutex;

#[unsafe(no_mangle)]
pub static _USES_DR_VERSION_: c_int = dynamorio_sys::_USES_DR_VERSION_;

struct Logger {
    file: File,
}

static LOGGER: Mutex<Option<Logger>> = Mutex::new(None);

#[unsafe(no_mangle)]
pub extern "C" fn dr_client_main(
    _id: client_id_t,
    _argc: c_int,
    _argv: *const *const c_char,
) {
    unsafe {
        dr_set_client_name(c"bblogger".as_ptr(), c"https://github.com/abscosmos/basic-block-trace".as_ptr());

        dr_register_exit_event(Some(exit_event));
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
use std::ffi::{c_char, c_int};
use dynamorio_sys::{client_id_t, dr_set_client_name};

#[unsafe(no_mangle)]
pub static _USES_DR_VERSION_: c_int = dynamorio_sys::_USES_DR_VERSION_;

#[unsafe(no_mangle)]
pub extern "C" fn dr_client_main(
    _id: client_id_t,
    _argc: c_int,
    _argv: *const *const c_char,
) {
    unsafe {
        dr_set_client_name(c"bblogger".as_ptr(), c"https://github.com/abscosmos/basic-block-trace".as_ptr());
    }
}
#[macro_export]
macro_rules! dr_println {
    ($($arg:tt)*) => {{
        unsafe {
            ::dynamorio_sys::dr_printf(
                ::std::ffi::CString::new(format!($($arg)*) + "\n")
                    .expect("shouldn't have null bytes")
                    .as_ptr()
            );
        }
    }};
}
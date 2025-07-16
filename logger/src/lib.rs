use std::borrow::Cow;
use std::ffi::{c_char, c_int, CStr};
use std::fs::File;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::slice;
use dynamorio_sys::{client_id_t, dr_free_module_data, dr_get_application_name, dr_get_main_module, dr_register_bb_event, dr_register_exit_event, dr_set_client_name, module_data_t};
use parking_lot::Mutex;

pub mod instruction;
mod event;

#[unsafe(no_mangle)]
pub static _USES_DR_VERSION_: c_int = dynamorio_sys::_USES_DR_VERSION_;

struct Logger {
    file: File,
    filter_module_addr: Option<NonZeroUsize>,
}

static LOGGER: Mutex<Option<Logger>> = Mutex::new(None);

/// # Safety
/// - `argc` and `argv` must be from main call
/// - `argv` should point to `argc` number of valid null terminated c strings
/// - the array pointed to by `argv` & the c strings should be
/// valid for the entire duration of the program (`'static`)
unsafe fn collect_args<'a>(argc: c_int, argv: *const *const c_char) -> impl Iterator<Item=Cow<'static, str>> + Clone {
    let args = unsafe { slice::from_raw_parts(argv, argc as _) };

    args.into_iter()
        .map(|&ptr|
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy()
        )
}

// TODO: replace this with some argument parsing library
fn get_arg_value<'a, T: AsRef<str> + 'a>(args: impl IntoIterator<Item=&'a T>, arg: &str) -> Option<&'a str> {
    let found = args.into_iter()
        .map(AsRef::as_ref)
        .find(|s| s.starts_with('-') && s[1..].starts_with(arg))?;

    let trim = &found[arg.len() + 1..];

    if trim.starts_with("=") {
        let val = &trim[1..];

        (!val.is_empty()).then_some(val)
    } else {
        trim.is_empty().then_some(trim)
    }
}

fn fallback_file_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("epoch must be before any current time")
        .as_secs();

    let name = match unsafe { dr_get_application_name() } {
        ptr if ptr.is_null() => "unknown".into(),
        ptr => unsafe { CStr::from_ptr(ptr) }.to_string_lossy(),
    };

    format!("{name}_trace-{timestamp:x}.log")
}

#[unsafe(no_mangle)]
pub extern "C" fn dr_client_main(
    _id: client_id_t,
    argc: c_int,
    argv: *const *const c_char,
) {
    unsafe {
        dr_set_client_name(c"bblogger".as_ptr(), c"https://github.com/abscosmos/basic-block-trace".as_ptr());

        dr_register_exit_event(Some(exit_event));

        dr_register_bb_event(Some(event::basic_block));
    }

    // SAFETY: args come from main, valid, static lifetime
    let args = unsafe { collect_args(argc, argv) }
        .collect::<Box<_>>();

    // TODO: sanitize file name
    let file_name = get_arg_value(&args, "file")
        .map(Cow::from)
        .unwrap_or_else(|| fallback_file_name().into());

    let filter = get_arg_value(&args, "filter") == Some("");

    let logger = Logger {
        file: File::create(file_name.as_ref()).expect("should be able to create file"),
        filter_module_addr: filter.then(main_module_start_addr),
    };

    println!("Saving logs to \"{file_name}\".");

    *LOGGER.lock() = Some(logger);
}

pub extern "C" fn exit_event() {
    // ensure Drop::drop runs for Logger
    let _ = LOGGER.lock().take();
}

fn main_module_start_addr() -> NonZeroUsize {
    let main_module = NonNull::new(unsafe { dr_get_main_module() })
        .expect("should be nonnull");

    let addr = unsafe { (*main_module.as_ptr()).__bindgen_anon_1.start.addr() };

    unsafe { dr_free_module_data(main_module.as_ptr()) };

    NonZeroUsize::new(addr).expect("addr couldn't have been nullptr")
}
use std::borrow::Cow;
use std::ffi::{c_char, c_int, CStr};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::{fs, slice};
use dynamorio_sys::{client_id_t, dr_free_module_data, dr_get_application_name, dr_get_main_module, dr_module_preferred_name, dr_printf, dr_register_bb_event, dr_register_exit_event, dr_set_client_name, module_data_t};
use hashbrown::HashMap;
use parking_lot::Mutex;
use logger_core::Application;
use crate::trace::TraceData;

pub mod trace;
pub mod instruction;
mod event;
mod log;

#[unsafe(no_mangle)]
pub static _USES_DR_VERSION_: c_int = dynamorio_sys::_USES_DR_VERSION_;

struct Logger {
    trace: TraceData,
    save_path: PathBuf,
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

fn fallback_file_name() -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("epoch must be before any current time")
        .as_secs();

    let name = match unsafe { dr_get_application_name() } {
        ptr if ptr.is_null() => "unknown".into(),
        ptr => unsafe { CStr::from_ptr(ptr) }.to_string_lossy(),
    };

    format!("{name}_trace-{timestamp:x}.trace").into()
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
        .map(|s| Cow::Borrowed(Path::new(s)))
        .unwrap_or_else(|| fallback_file_name().into());

    let filter = get_arg_value(&args, "filter") == Some("");

    let main_module_raw = NonNull::new(unsafe { dr_get_main_module() })
        .expect("should be nonnull");

    let target_application = module_to_application(unsafe { main_module_raw.as_ref() });

    unsafe { dr_free_module_data(main_module_raw.as_ptr()); }

    let logger = Logger {
        trace: TraceData {
            targeted: target_application,
            blocks: HashMap::default(),
            filter,
        },
        save_path: file_name.as_ref().into(),
    };

    *LOGGER.lock() = Some(logger);
}

pub extern "C" fn exit_event() {
    // ensure Drop::drop runs for Logger
    let Logger { trace, save_path } = LOGGER.lock()
        .take()
        .expect("logger should be initialized");

    let binary = postcard::to_allocvec(&trace)
        .expect("trace data should be safely serializable");

    fs::write(&save_path, binary)
        .expect("should be able to write to file");

    dr_println!("Saved trace to \"{}\".", save_path.display());
    dr_println!("Summary:\n{:#?}", trace.summary());
}

/// this does not semantically take ownership of `module`,
/// and as such will not free it
pub fn module_to_application(module: &module_data_t) -> Application {
    let name = match unsafe { dr_module_preferred_name(module) } {
        ptr if ptr.is_null() => "{unknown}".into(),
        ptr => unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into(),
    };

    let address = unsafe { module.__bindgen_anon_1.start }
        .addr()
        .try_into()
        .expect("module start addr shouldn't be nullptr");

    Application { name, address }
}
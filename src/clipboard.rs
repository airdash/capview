use anyhow::{bail, Result};
use std::io::Write;
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// Global handle to the current clipboard-serving thread.
/// When a new copy replaces it, the old thread is signalled to stop.
static CLIPBOARD_THREAD: Mutex<Option<ClipboardHandle>> = Mutex::new(None);

struct ClipboardHandle {
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for ClipboardHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
            loop {
                if t.is_finished() {
                    let _ = t.join();
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    drop(t);
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
}

/// Copy PNG data to Wayland clipboard.
/// Spawns a background thread that keeps the data source alive so
/// other apps can paste from us. Tries ext-data-control-v1 first,
/// falls back to wlr-data-control-unstable-v1.
pub fn copy_to_clipboard(png_data: &[u8], debug: bool) -> Result<()> {
    if debug { eprintln!("debug: clipboard: {}B png", png_data.len()); }

    let proto = detect_protocol(debug)?;
    if debug { eprintln!("debug: clipboard: using {:?}", proto); }

    let data = png_data.to_vec();
    let running = Arc::new(AtomicBool::new(true));
    let running2 = running.clone();

    let handle = thread::spawn(move || {
        crate::priority::avoid_render_core();
        if let Err(e) = serve_clipboard(&data, proto, &running2, debug) {
            eprintln!("clipboard: {}", e);
        }
    });

    // Replace previous clipboard thread (compositor will cancel its source)
    let new_handle = ClipboardHandle {
        running,
        thread: Some(handle),
    };
    if let Ok(mut guard) = CLIPBOARD_THREAD.lock() {
        *guard = Some(new_handle);
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum Proto { Ext, Wlr }

/// Quick probe: connect, check globals, disconnect.
fn detect_protocol(debug: bool) -> Result<Proto> {
    use wayland_client::globals::registry_queue_init;
    use wayland_client::Connection;

    let conn = Connection::connect_to_env()?;

    struct Probe;
    impl wayland_client::Dispatch<wayland_client::protocol::wl_registry::WlRegistry,
                                   wayland_client::globals::GlobalListContents> for Probe {
        fn event(_: &mut Self, _: &wayland_client::protocol::wl_registry::WlRegistry,
                 _: wayland_client::protocol::wl_registry::Event,
                 _: &wayland_client::globals::GlobalListContents,
                 _: &Connection, _: &wayland_client::QueueHandle<Self>) {}
    }

    let (globals, _queue) = registry_queue_init::<Probe>(&conn)?;

    let has_ext = globals.contents().clone_list().iter()
        .any(|g| g.interface == "ext_data_control_manager_v1");
    let has_wlr = globals.contents().clone_list().iter()
        .any(|g| g.interface == "zwlr_data_control_manager_v1");

    if debug { eprintln!("debug: clipboard: ext={} wlr={}", has_ext, has_wlr); }

    if has_ext {
        Ok(Proto::Ext)
    } else if has_wlr {
        Ok(Proto::Wlr)
    } else {
        bail!("compositor supports neither ext-data-control-v1 nor wlr-data-control-v1")
    }
}

fn serve_clipboard(data: &[u8], proto: Proto, running: &AtomicBool, debug: bool) -> Result<()> {
    match proto {
        Proto::Ext => serve_ext(data, running, debug),
        Proto::Wlr => serve_wlr(data, running, debug),
    }
}

// ── ext-data-control-v1 ─────────────────────────────────────────────

fn serve_ext(data: &[u8], running: &AtomicBool, debug: bool) -> Result<()> {
    use wayland_client::globals::{registry_queue_init, GlobalListContents};
    use wayland_client::protocol::{wl_registry, wl_seat};
    use wayland_client::{Connection, Dispatch, QueueHandle};
    use wayland_protocols::ext::data_control::v1::client::{
        ext_data_control_device_v1::ExtDataControlDeviceV1,
        ext_data_control_manager_v1::ExtDataControlManagerV1,
        ext_data_control_offer_v1::ExtDataControlOfferV1,
        ext_data_control_source_v1::{self, ExtDataControlSourceV1},
    };

    struct State { data: Vec<u8>, done: bool, send_count: u32, debug: bool }

    impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
        fn event(_: &mut Self, _: &wl_registry::WlRegistry,
                 _: wl_registry::Event, _: &GlobalListContents,
                 _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<wl_seat::WlSeat, ()> for State {
        fn event(_: &mut Self, _: &wl_seat::WlSeat, _: wl_seat::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<ExtDataControlManagerV1, ()> for State {
        fn event(_: &mut Self, _: &ExtDataControlManagerV1,
                 _: <ExtDataControlManagerV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<ExtDataControlDeviceV1, ()> for State {
        fn event(_: &mut Self, _: &ExtDataControlDeviceV1,
                 _: <ExtDataControlDeviceV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
        wayland_client::event_created_child!(State, ExtDataControlDeviceV1, [
            0 => (ExtDataControlOfferV1, ()),
        ]);
    }
    impl Dispatch<ExtDataControlOfferV1, ()> for State {
        fn event(_: &mut Self, _: &ExtDataControlOfferV1,
                 _: <ExtDataControlOfferV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<ExtDataControlSourceV1, ()> for State {
        fn event(state: &mut Self, _: &ExtDataControlSourceV1,
                 event: <ExtDataControlSourceV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {
            match event {
                ext_data_control_source_v1::Event::Send { fd, mime_type } => {
                    state.send_count += 1;
                    if state.debug { eprintln!("debug: clipboard ext: send #{} mime={}", state.send_count, mime_type); }
                    // Use into_raw_fd() to consume the OwnedFd, preventing double-close
                    let mut f = unsafe { std::fs::File::from_raw_fd(fd.into_raw_fd()) };
                    match f.write_all(&state.data) {
                        Ok(()) => { if state.debug { eprintln!("debug: clipboard ext: wrote {}B", state.data.len()); } }
                        Err(e) => eprintln!("clipboard ext: write error: {}", e),
                    }
                }
                ext_data_control_source_v1::Event::Cancelled => {
                    if state.debug { eprintln!("debug: clipboard ext: cancelled (served {} pastes)", state.send_count); }
                    state.done = true;
                }
                _ => {}
            }
        }
    }

    let conn = Connection::connect_to_env()?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn)?;
    let qh = queue.handle();
    let mut state = State { data: data.to_vec(), done: false, send_count: 0, debug };

    let manager: ExtDataControlManagerV1 = globals.bind(&qh, 1..=1, ())
        .map_err(|_| anyhow::anyhow!("ext-data-control-v1 not available"))?;
    let seat: wl_seat::WlSeat = globals.bind(&qh, 1..=9, ())
        .map_err(|_| anyhow::anyhow!("no wl_seat"))?;

    let source = manager.create_data_source(&qh, ());
    source.offer("image/png".into());
    let device = manager.get_data_device(&seat, &qh, ());
    device.set_selection(Some(&source));

    if debug { eprintln!("debug: clipboard ext: serving (thread alive until cancelled)"); }

    while !state.done && running.load(Ordering::Relaxed) {
        queue.flush()?;
        let read_guard = queue.prepare_read().unwrap();
        let fd = read_guard.connection_fd();
        let mut pfd = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        let ret = unsafe { libc::poll(&mut pfd, 1, 250) };
        if ret > 0 { let _ = read_guard.read(); } else { drop(read_guard); }
        queue.dispatch_pending(&mut state)?;
    }

    if debug { eprintln!("debug: clipboard ext: thread exiting"); }
    Ok(())
}

// ── wlr-data-control-unstable-v1 ────────────────────────────────────

fn serve_wlr(data: &[u8], running: &AtomicBool, debug: bool) -> Result<()> {
    use wayland_client::globals::{registry_queue_init, GlobalListContents};
    use wayland_client::protocol::{wl_registry, wl_seat};
    use wayland_client::{Connection, Dispatch, QueueHandle};
    use wayland_protocols_wlr::data_control::v1::client::{
        zwlr_data_control_device_v1::ZwlrDataControlDeviceV1,
        zwlr_data_control_manager_v1::ZwlrDataControlManagerV1,
        zwlr_data_control_offer_v1::ZwlrDataControlOfferV1,
        zwlr_data_control_source_v1::{self, ZwlrDataControlSourceV1},
    };

    struct State { data: Vec<u8>, done: bool, send_count: u32, debug: bool }

    impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
        fn event(_: &mut Self, _: &wl_registry::WlRegistry,
                 _: wl_registry::Event, _: &GlobalListContents,
                 _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<wl_seat::WlSeat, ()> for State {
        fn event(_: &mut Self, _: &wl_seat::WlSeat, _: wl_seat::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<ZwlrDataControlManagerV1, ()> for State {
        fn event(_: &mut Self, _: &ZwlrDataControlManagerV1,
                 _: <ZwlrDataControlManagerV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<ZwlrDataControlDeviceV1, ()> for State {
        fn event(_: &mut Self, _: &ZwlrDataControlDeviceV1,
                 _: <ZwlrDataControlDeviceV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
        wayland_client::event_created_child!(State, ZwlrDataControlDeviceV1, [
            0 => (ZwlrDataControlOfferV1, ()),
        ]);
    }
    impl Dispatch<ZwlrDataControlOfferV1, ()> for State {
        fn event(_: &mut Self, _: &ZwlrDataControlOfferV1,
                 _: <ZwlrDataControlOfferV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {}
    }
    impl Dispatch<ZwlrDataControlSourceV1, ()> for State {
        fn event(state: &mut Self, _: &ZwlrDataControlSourceV1,
                 event: <ZwlrDataControlSourceV1 as wayland_client::Proxy>::Event,
                 _: &(), _: &Connection, _: &QueueHandle<Self>) {
            match event {
                zwlr_data_control_source_v1::Event::Send { fd, mime_type } => {
                    state.send_count += 1;
                    if state.debug { eprintln!("debug: clipboard wlr: send #{} mime={}", state.send_count, mime_type); }
                    // Use into_raw_fd() to consume the OwnedFd, preventing double-close
                    let mut f = unsafe { std::fs::File::from_raw_fd(fd.into_raw_fd()) };
                    match f.write_all(&state.data) {
                        Ok(()) => { if state.debug { eprintln!("debug: clipboard wlr: wrote {}B", state.data.len()); } }
                        Err(e) => eprintln!("clipboard wlr: write error: {}", e),
                    }
                }
                zwlr_data_control_source_v1::Event::Cancelled => {
                    if state.debug { eprintln!("debug: clipboard wlr: cancelled (served {} pastes)", state.send_count); }
                    state.done = true;
                }
                _ => {}
            }
        }
    }

    let conn = Connection::connect_to_env()?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn)?;
    let qh = queue.handle();
    let mut state = State { data: data.to_vec(), done: false, send_count: 0, debug };

    let manager: ZwlrDataControlManagerV1 = globals.bind(&qh, 1..=2, ())
        .map_err(|_| anyhow::anyhow!("wlr-data-control-v1 not available"))?;
    let seat: wl_seat::WlSeat = globals.bind(&qh, 1..=9, ())
        .map_err(|_| anyhow::anyhow!("no wl_seat"))?;

    let source = manager.create_data_source(&qh, ());
    source.offer("image/png".into());
    let device = manager.get_data_device(&seat, &qh, ());
    device.set_selection(Some(&source));

    if debug { eprintln!("debug: clipboard wlr: serving (thread alive until cancelled)"); }

    while !state.done && running.load(Ordering::Relaxed) {
        queue.flush()?;
        let read_guard = queue.prepare_read().unwrap();
        let fd = read_guard.connection_fd();
        let mut pfd = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        let ret = unsafe { libc::poll(&mut pfd, 1, 250) };
        if ret > 0 { let _ = read_guard.read(); } else { drop(read_guard); }
        queue.dispatch_pending(&mut state)?;
    }

    if debug { eprintln!("debug: clipboard wlr: thread exiting"); }
    Ok(())
}

/// Check whether we're running under Wayland.
pub fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

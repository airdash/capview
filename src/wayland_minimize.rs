use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{wl_output, wl_registry};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle};
use wayland_protocols_wlr::foreign_toplevel::v1::client::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};

/// Watches window minimize state on Wayland via wlr-foreign-toplevel-management.
/// Works on Sway, wlroots compositors, and KDE Plasma 6+.
/// Returns None on compositors without the protocol (e.g. KDE Plasma 5.x),
/// in which case main.rs falls back to treating FocusLost as minimize.
pub struct MinimizeWatcher {
    minimized: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    _thread: thread::JoinHandle<()>,
}

impl MinimizeWatcher {
    pub fn start(debug: bool) -> Option<Self> {
        let minimized = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));
        let minimized2 = minimized.clone();
        let running2 = running.clone();

        let handle = thread::spawn(move || {
            crate::priority::avoid_render_core();
            if let Err(e) = watch_loop(&minimized2, &running2, debug) {
                if debug { eprintln!("debug: wayland-minimize: {}", e); }
            }
        });

        // Give the thread time to connect and check protocol availability
        thread::sleep(std::time::Duration::from_millis(100));
        if !running.load(Ordering::Relaxed) {
            let _ = handle.join();
            return None;
        }

        Some(Self {
            minimized,
            running,
            _thread: handle,
        })
    }

    pub fn is_minimized(&self) -> bool {
        self.minimized.load(Ordering::Relaxed)
    }
}

impl Drop for MinimizeWatcher {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

struct State {
    minimized: Arc<AtomicBool>,
    debug: bool,
    toplevels: Vec<ToplevelInfo>,
}

struct ToplevelInfo {
    id: wayland_client::backend::ObjectId,
    app_id: String,
    is_minimized: bool,
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut Self, _: &wl_registry::WlRegistry, _: wl_registry::Event,
        _: &GlobalListContents, _: &Connection, _: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<wl_output::WlOutput, ()> for State {
    fn event(
        _: &mut Self, _: &wl_output::WlOutput, _: wl_output::Event,
        _: &(), _: &Connection, _: &QueueHandle<Self>,
    ) {}
}

impl Dispatch<ZwlrForeignToplevelManagerV1, ()> for State {
    fn event(
        _: &mut Self, _: &ZwlrForeignToplevelManagerV1,
        event: zwlr_foreign_toplevel_manager_v1::Event,
        _: &(), _: &Connection, _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_foreign_toplevel_manager_v1::Event::Toplevel { .. } => {}
            zwlr_foreign_toplevel_manager_v1::Event::Finished => {}
            _ => {}
        }
    }

    wayland_client::event_created_child!(State, ZwlrForeignToplevelManagerV1, [
        0 => (ZwlrForeignToplevelHandleV1, ()),
    ]);
}

impl Dispatch<ZwlrForeignToplevelHandleV1, ()> for State {
    fn event(
        state: &mut Self, proxy: &ZwlrForeignToplevelHandleV1,
        event: zwlr_foreign_toplevel_handle_v1::Event,
        _: &(), _: &Connection, _: &QueueHandle<Self>,
    ) {
        let obj_id = proxy.id();

        let ensure_entry = |toplevels: &mut Vec<ToplevelInfo>, id: &wayland_client::backend::ObjectId| {
            if toplevels.iter().all(|t| t.id != *id) {
                toplevels.push(ToplevelInfo {
                    id: id.clone(),
                    app_id: String::new(),
                    is_minimized: false,
                });
            }
        };

        match event {
            zwlr_foreign_toplevel_handle_v1::Event::AppId { app_id } => {
                ensure_entry(&mut state.toplevels, &obj_id);
                if let Some(info) = state.toplevels.iter_mut().find(|t| t.id == obj_id) {
                    info.app_id = app_id;
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Title { title } => {
                ensure_entry(&mut state.toplevels, &obj_id);
                if let Some(info) = state.toplevels.iter_mut().find(|t| t.id == obj_id) {
                    if info.app_id.is_empty() && title.starts_with("capview") {
                        info.app_id = "capview".into();
                    }
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::State { state: raw_state } => {
                let states: Vec<u32> = raw_state
                    .chunks_exact(4)
                    .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let minimized = states.contains(&1);

                ensure_entry(&mut state.toplevels, &obj_id);
                if let Some(info) = state.toplevels.iter_mut().find(|t| t.id == obj_id) {
                    info.is_minimized = minimized;
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Done => {
                let any_minimized = state.toplevels.iter().any(|t| {
                    t.app_id == "capview" && t.is_minimized
                });
                let prev = state.minimized.swap(any_minimized, Ordering::SeqCst);
                if state.debug && prev != any_minimized {
                    eprintln!("debug: wayland-minimize: capview minimized={}", any_minimized);
                }
            }
            zwlr_foreign_toplevel_handle_v1::Event::Closed => {
                state.toplevels.retain(|t| t.id != obj_id);
                let any_minimized = state.toplevels.iter().any(|t| {
                    t.app_id == "capview" && t.is_minimized
                });
                state.minimized.store(any_minimized, Ordering::SeqCst);
            }
            _ => {}
        }
    }
}

fn watch_loop(
    minimized: &Arc<AtomicBool>,
    running: &Arc<AtomicBool>,
    debug: bool,
) -> anyhow::Result<()> {
    let conn = Connection::connect_to_env()?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn)?;
    let qh = queue.handle();

    let has_ftm = globals.contents().clone_list().iter()
        .any(|g| g.interface == "zwlr_foreign_toplevel_manager_v1");

    if !has_ftm {
        if debug { eprintln!("debug: wayland-minimize: zwlr_foreign_toplevel_manager_v1 not available, using FocusLost fallback"); }
        running.store(false, Ordering::SeqCst);
        return Ok(());
    }

    let _manager: ZwlrForeignToplevelManagerV1 = globals
        .bind(&qh, 1..=3, ())
        .map_err(|_| anyhow::anyhow!("failed to bind zwlr_foreign_toplevel_manager_v1"))?;

    let mut state = State {
        minimized: minimized.clone(),
        debug,
        toplevels: Vec::new(),
    };

    if debug { eprintln!("debug: wayland-minimize: watching foreign toplevels"); }

    while running.load(Ordering::Relaxed) {
        queue.flush()?;
        let read_guard = queue.prepare_read().unwrap();
        let fd = read_guard.connection_fd();
        let mut pfd = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ret = unsafe { libc::poll(&mut pfd, 1, 250) };
        if ret > 0 {
            let _ = read_guard.read();
        } else {
            drop(read_guard);
        }
        queue.dispatch_pending(&mut state)?;
    }

    Ok(())
}

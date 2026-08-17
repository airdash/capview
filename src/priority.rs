/// Process and thread priority optimizations for lowest possible latency.
/// All operations are best-effort — failures are logged and ignored.

use crate::config::PriorityFlags;

/// Apply selected priority optimizations for the main (render) thread.
/// Call once, early in startup, before entering the main loop.
#[cfg(target_os = "linux")]
pub fn apply_all(flags: PriorityFlags, debug: bool) {
    if flags.has(PriorityFlags::TIMER_SLACK) { set_timer_slack(debug); }
    if flags.has(PriorityFlags::REALTIME) { try_realtime_scheduling(debug); }
    if flags.has(PriorityFlags::CPU_PIN) { pin_current_thread(debug); }
    if flags.has(PriorityFlags::MLOCK) { mlock_current(debug); }
    if flags.has(PriorityFlags::IO_PRIO) { set_io_priority(debug); }
    if flags.has(PriorityFlags::SIG_MASK) { mask_render_signals(debug); }
}

#[cfg(target_os = "macos")]
pub fn apply_all(flags: PriorityFlags, debug: bool) {
    if flags.has(PriorityFlags::REALTIME) { try_realtime_scheduling(debug); }
    if flags.has(PriorityFlags::MLOCK) { mlock_current(debug); }
    if flags.has(PriorityFlags::SIG_MASK) { mask_render_signals(debug); }
}

// ── PR_SET_TIMERSLACK (Linux) ───────────────────────────────────────────

#[cfg(target_os = "linux")]
fn set_timer_slack(debug: bool) {
    // Default kernel timer slack is 50µs.  Setting to 1ns makes ppoll/nanosleep
    // wake up as tightly as the scheduler allows.
    const PR_SET_TIMERSLACK: libc::c_int = 29;
    let ret = unsafe { libc::prctl(PR_SET_TIMERSLACK, 1u64, 0u64, 0u64, 0u64) };
    if ret == 0 {
        if debug { eprintln!("priority: timer slack set to 1ns"); }
    } else {
        eprintln!("priority: PR_SET_TIMERSLACK failed: {}", std::io::Error::last_os_error());
    }
}

// ── Real-time scheduling ────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn try_realtime_scheduling(debug: bool) {
    // Try SCHED_RR with minimum RT priority.  Succeeds without root if
    // RLIMIT_RTPRIO allows it (common on audio-oriented distros / pipewire setups).
    let param = libc::sched_param { sched_priority: 1 };
    let ret = unsafe { libc::sched_setscheduler(0, libc::SCHED_RR, &param) };
    if ret == 0 {
        if debug { eprintln!("priority: SCHED_RR enabled (priority=1)"); }
    } else {
        let e = std::io::Error::last_os_error();
        if debug { eprintln!("priority: SCHED_RR unavailable ({}), using default scheduler", e); }
    }
}

#[cfg(target_os = "macos")]
fn try_realtime_scheduling(debug: bool) {
    // macOS: request time-constraint (real-time) thread policy via Mach.
    // This tells the scheduler we need periodic, low-latency wakeups.
    use std::ffi::c_uint;

    extern "C" {
        fn mach_thread_self() -> c_uint;
        fn thread_policy_set(
            thread: c_uint,
            flavor: c_uint,
            policy_info: *const c_uint,
            count: c_uint,
        ) -> i32;
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    }

    #[repr(C)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    const THREAD_TIME_CONSTRAINT_POLICY: c_uint = 2;
    // thread_time_constraint_policy has 4 u32 fields: period, computation, constraint, preemptible
    // Target: 8.3ms period (120fps), 2ms computation, 4ms constraint
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    if unsafe { mach_timebase_info(&mut info) } != 0 {
        if debug { eprintln!("priority: mach_timebase_info failed"); }
        return;
    }
    let ns_to_abs = |ns: u64| -> u32 { (ns * info.denom as u64 / info.numer as u64) as u32 };
    let policy: [c_uint; 4] = [
        ns_to_abs(8_333_333),  // period (~120fps)
        ns_to_abs(2_000_000),  // computation (2ms)
        ns_to_abs(4_000_000),  // constraint (4ms)
        1,                     // preemptible
    ];
    let ret = unsafe {
        thread_policy_set(mach_thread_self(), THREAD_TIME_CONSTRAINT_POLICY, policy.as_ptr(), 4)
    };
    if ret == 0 {
        if debug { eprintln!("priority: macOS time-constraint thread policy set"); }
    } else {
        if debug { eprintln!("priority: macOS thread_policy_set failed ({})", ret); }
    }
}

// ── CPU affinity (Linux) ────────────────────────────────────────────────

/// Number of online CPUs (cached).
#[cfg(target_os = "linux")]
fn online_cpus() -> usize {
    unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN).max(1) as usize }
}

/// Sample per-CPU idle ticks from /proc/stat.  Returns (cpu_index, idle_ticks)
/// sorted by most-idle-first.  Two samples 50ms apart give a usage snapshot.
#[cfg(target_os = "linux")]
fn rank_cpus_by_idle() -> Vec<usize> {
    fn read_cpu_ticks() -> Vec<(usize, u64, u64)> {
        // Returns (cpu_index, total_ticks, idle_ticks) for each cpu line
        let stat = match std::fs::read_to_string("/proc/stat") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let mut cpus = Vec::new();
        for line in stat.lines() {
            if !line.starts_with("cpu") { continue; }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 5 { continue; }
            // Skip aggregate "cpu" line, only want "cpu0", "cpu1", etc.
            let name = fields[0];
            if name == "cpu" { continue; }
            let idx: usize = match name.strip_prefix("cpu").and_then(|s| s.parse().ok()) {
                Some(i) => i,
                None => continue,
            };
            let ticks: Vec<u64> = fields[1..].iter().filter_map(|s| s.parse().ok()).collect();
            let total: u64 = ticks.iter().sum();
            let idle = ticks.get(3).copied().unwrap_or(0); // 4th field = idle
            cpus.push((idx, total, idle));
        }
        cpus
    }

    let before = read_cpu_ticks();
    if before.is_empty() { return Vec::new(); }
    std::thread::sleep(std::time::Duration::from_millis(25));
    let after = read_cpu_ticks();

    // Compute idle fraction delta for each CPU
    let mut idle_fracs: Vec<(usize, f64)> = Vec::new();
    for (idx, total_a, idle_a) in &after {
        if let Some((_, total_b, idle_b)) = before.iter().find(|(i, _, _)| i == idx) {
            let dt = total_a.saturating_sub(*total_b);
            let di = idle_a.saturating_sub(*idle_b);
            let frac = if dt > 0 { di as f64 / dt as f64 } else { 0.0 };
            idle_fracs.push((*idx, frac));
        }
    }
    // Sort by most idle first; break ties by preferring higher-numbered cores
    // (lower cores tend to catch more IRQs and system tasks)
    idle_fracs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            .then(b.0.cmp(&a.0))
    });
    idle_fracs.iter().map(|(idx, _)| *idx).collect()
}

/// Directory for per-CPU lock files so concurrent capview instances avoid each
/// other's render cores.  Uses $XDG_RUNTIME_DIR/capview/ (typically /run/user/UID/capview/).
#[cfg(target_os = "linux")]
fn lock_dir() -> std::path::PathBuf {
    let base = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| format!("/tmp/capview-{}", unsafe { libc::getuid() }));
    std::path::PathBuf::from(base).join("capview")
}

/// Try to acquire an exclusive flock on a per-CPU lock file.
/// Returns the held File on success (caller must keep it alive).
#[cfg(target_os = "linux")]
fn try_lock_cpu(cpu: usize) -> Option<std::fs::File> {
    use std::os::unix::io::AsRawFd;
    let dir = lock_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("cpu_{}", cpu));
    let f = std::fs::OpenOptions::new().create(true).write(true).open(&path).ok()?;
    let ret = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 { Some(f) } else { None }
}

/// Returns the set of CPUs currently locked by other capview instances.
#[cfg(target_os = "linux")]
fn cpus_locked_by_others(ncpus: usize) -> Vec<usize> {
    use std::os::unix::io::AsRawFd;
    let dir = lock_dir();
    let mut locked = Vec::new();
    for cpu in 0..ncpus {
        let path = dir.join(format!("cpu_{}", cpu));
        if let Ok(f) = std::fs::OpenOptions::new().read(true).open(&path) {
            let ret = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if ret != 0 {
                // Someone else holds it
                locked.push(cpu);
            } else {
                // We got it — unlock immediately, we were just probing
                unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
            }
        }
    }
    locked
}

#[cfg(target_os = "linux")]
fn pin_current_thread(debug: bool) {
    let ncpus = online_cpus();
    if ncpus < 2 {
        if debug { eprintln!("priority: single CPU, skipping affinity"); }
        return;
    }
    // Rank cores by idle time, prefer high-numbered cores.
    let ranked = rank_cpus_by_idle();
    // Find which CPUs other capview instances have claimed.
    let others = cpus_locked_by_others(ncpus);
    if debug && !others.is_empty() {
        eprintln!("priority: CPUs claimed by other capview instances: {:?}", others);
    }
    // Pick the first ranked core not claimed by another instance.
    let render_cpu = ranked.iter()
        .find(|cpu| !others.contains(cpu))
        .or(ranked.first())
        .copied()
        .unwrap_or(0);

    // Acquire lock on chosen CPU (held for process lifetime via static).
    let lock = try_lock_cpu(render_cpu);
    if lock.is_none() && debug {
        eprintln!("priority: could not lock CPU {} (contention), pinning anyway", render_cpu);
    }

    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::CPU_ZERO(&mut set) };
    unsafe { libc::CPU_SET(render_cpu, &mut set) };
    let ret = unsafe {
        libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        )
    };
    if ret == 0 {
        RENDER_CPU.store(render_cpu as i32, std::sync::atomic::Ordering::Relaxed);
        // Pre-compute the background thread cpu_set (excludes all locked render cores).
        // Build in a local, then publish to the static via a raw pointer to avoid
        // creating a `&mut` reference to a mutable static (Rust 2024 lint).
        let mut bg_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let mut count = 0usize;
        unsafe { libc::CPU_ZERO(&mut bg_set); }
        for cpu in 0..ncpus {
            if !others.contains(&cpu) {
                unsafe { libc::CPU_SET(cpu, &mut bg_set); }
                count += 1;
            }
        }
        unsafe {
            std::ptr::write(std::ptr::addr_of_mut!(BG_CPU_SET), bg_set);
            std::ptr::write(std::ptr::addr_of_mut!(BG_SET_COUNT), count);
        }
        CPU_PIN_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        // Store lock to keep it held for process lifetime
        unsafe { CPU_LOCK = lock; }
        if debug { eprintln!("priority: render thread pinned to CPU {} (most idle, avoiding {:?})", render_cpu, others); }
    } else {
        if debug { eprintln!("priority: CPU affinity failed ({})", std::io::Error::from_raw_os_error(ret)); }
    }
}

/// Held for process lifetime to signal our render CPU to other instances.
#[cfg(target_os = "linux")]
static mut CPU_LOCK: Option<std::fs::File> = None;

/// Call from background threads to keep them off all capview render cores.
/// Uses a cached cpu_set computed at init time — no syscalls per call.
#[cfg(target_os = "linux")]
pub fn avoid_render_core() {
    if !CPU_PIN_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) { return; }
    unsafe {
        if BG_SET_COUNT == 0 { return; }
        libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            std::ptr::addr_of!(BG_CPU_SET),
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn avoid_render_core() {}

/// Cached cpu_set for background threads (excludes all render cores).
/// Computed once in pin_current_thread(), then reused by avoid_render_core().
#[cfg(target_os = "linux")]
static mut BG_CPU_SET: libc::cpu_set_t = unsafe { std::mem::zeroed() };
#[cfg(target_os = "linux")]
static mut BG_SET_COUNT: usize = 0;

/// Set once by pin_current_thread() so background threads know to avoid it.
#[cfg(target_os = "linux")]
static CPU_PIN_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(target_os = "linux")]
static RENDER_CPU: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

// ── CPU frequency monitoring (Linux) ─────────────────────────────────────

/// Read the current frequency (kHz) of the render CPU.
/// Returns 0 if unavailable (no cpufreq, permission denied, etc.).
#[cfg(target_os = "linux")]
pub fn render_cpu_freq_khz() -> u64 {
    let cpu = RENDER_CPU.load(std::sync::atomic::Ordering::Relaxed);
    if cpu < 0 { return 0; }
    cpu_freq_khz(cpu as usize)
}

/// Read the max frequency (kHz) of the render CPU.
#[cfg(target_os = "linux")]
pub fn render_cpu_max_freq_khz() -> u64 {
    let cpu = RENDER_CPU.load(std::sync::atomic::Ordering::Relaxed);
    if cpu < 0 { return 0; }
    let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", cpu);
    std::fs::read_to_string(&path).ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn cpu_freq_khz(cpu: usize) -> u64 {
    let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", cpu);
    std::fs::read_to_string(&path).ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

#[cfg(not(target_os = "linux"))]
pub fn render_cpu_freq_khz() -> u64 { 0 }
#[cfg(not(target_os = "linux"))]
pub fn render_cpu_max_freq_khz() -> u64 { 0 }

/// Which CPU the render thread is pinned to (-1 if not pinned).
pub fn render_cpu_id() -> i32 {
    #[cfg(target_os = "linux")]
    { RENDER_CPU.load(std::sync::atomic::Ordering::Relaxed) }
    #[cfg(not(target_os = "linux"))]
    { -1 }
}

/// Check if the render CPU is thermally throttled (frequency dropped below
/// 80% of max).  Returns (current_mhz, max_mhz, is_throttled).
#[cfg(target_os = "linux")]
pub fn check_throttle() -> (u32, u32, bool) {
    let cur = render_cpu_freq_khz();
    let max = render_cpu_max_freq_khz();
    let cur_mhz = (cur / 1000) as u32;
    let max_mhz = (max / 1000) as u32;
    let throttled = max > 0 && cur > 0 && cur < max * 80 / 100;
    (cur_mhz, max_mhz, throttled)
}

#[cfg(not(target_os = "linux"))]
pub fn check_throttle() -> (u32, u32, bool) { (0, 0, false) }

/// Re-evaluate CPU affinity: if the current render core is throttled,
/// find a cooler core and migrate.  Returns the new CPU id if re-pinned.
#[cfg(target_os = "linux")]
pub fn repin_if_throttled(debug: bool) -> Option<usize> {
    if !CPU_PIN_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) { return None; }
    let (cur_mhz, max_mhz, throttled) = check_throttle();
    if !throttled { return None; }

    if debug {
        eprintln!("priority: render CPU {} throttled ({}/{}MHz), re-evaluating",
            RENDER_CPU.load(std::sync::atomic::Ordering::Relaxed), cur_mhz, max_mhz);
    }

    let ncpus = online_cpus();
    let ranked = rank_cpus_by_idle();
    let current = RENDER_CPU.load(std::sync::atomic::Ordering::Relaxed) as usize;
    let others = cpus_locked_by_others(ncpus);

    // Find a core that isn't throttled, isn't the current one, and isn't locked
    let new_cpu = ranked.iter()
        .filter(|&&cpu| cpu != current && !others.contains(&cpu))
        .find(|&&cpu| {
            let freq = cpu_freq_khz(cpu);
            let max = {
                let path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", cpu);
                std::fs::read_to_string(&path).ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(0)
            };
            max == 0 || freq >= max * 85 / 100 // prefer cores running at >= 85% of max
        })
        .or_else(|| ranked.iter().find(|&&cpu| cpu != current && !others.contains(&cpu)))
        .copied();

    let new_cpu = match new_cpu {
        Some(c) => c,
        None => { return None; }
    };

    // Release old lock, acquire new one
    unsafe { CPU_LOCK = try_lock_cpu(new_cpu); }

    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::CPU_ZERO(&mut set) };
    unsafe { libc::CPU_SET(new_cpu, &mut set) };
    let ret = unsafe {
        libc::pthread_setaffinity_np(
            libc::pthread_self(),
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        )
    };
    if ret == 0 {
        RENDER_CPU.store(new_cpu as i32, std::sync::atomic::Ordering::Relaxed);
        if debug {
            let new_freq = cpu_freq_khz(new_cpu) / 1000;
            eprintln!("priority: re-pinned render thread to CPU {} ({}MHz)", new_cpu, new_freq);
        }
        Some(new_cpu)
    } else {
        if debug { eprintln!("priority: re-pin to CPU {} failed", new_cpu); }
        None
    }
}

#[cfg(not(target_os = "linux"))]
pub fn repin_if_throttled(_debug: bool) -> Option<usize> { None }

// ── mlock capture buffers ───────────────────────────────────────────────

fn mlock_current(debug: bool) {
    // Lock all current AND future pages into RAM.  MCL_FUTURE prevents page
    // faults on new allocations (OSD textures, frame buffers, etc.).
    // Respects RLIMIT_MEMLOCK (usually 64MB+, plenty for frame buffers).
    let ret = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };
    if ret == 0 {
        if debug { eprintln!("priority: mlockall(MCL_CURRENT|MCL_FUTURE) succeeded"); }
    } else {
        // MCL_FUTURE may be denied if RLIMIT_MEMLOCK is tight — fall back to MCL_CURRENT.
        let e = std::io::Error::last_os_error();
        if debug { eprintln!("priority: mlockall with MCL_FUTURE failed ({}), trying MCL_CURRENT only", e); }
        let ret2 = unsafe { libc::mlockall(libc::MCL_CURRENT) };
        if ret2 == 0 {
            if debug { eprintln!("priority: mlockall(MCL_CURRENT) succeeded"); }
        } else {
            let e2 = std::io::Error::last_os_error();
            if debug { eprintln!("priority: mlockall failed ({}), pages may be swapped", e2); }
        }
    }
}

// ── I/O priority (Linux) ────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn set_io_priority(debug: bool) {
    // ioprio_set(IOPRIO_WHO_PROCESS, 0 = self, class | priority)
    // Class 1 = IOPRIO_CLASS_RT needs CAP_SYS_ADMIN.
    // Class 2 = IOPRIO_CLASS_BE (best-effort), priority 0 = highest. No root needed.
    const IOPRIO_WHO_PROCESS: libc::c_int = 1;
    const IOPRIO_CLASS_BE: u32 = 2;
    const IOPRIO_PRIO_VALUE: u32 = (IOPRIO_CLASS_BE << 13) | 0; // class=BE, level=0
    let ret = unsafe {
        libc::syscall(libc::SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0, IOPRIO_PRIO_VALUE)
    };
    if ret == 0 {
        if debug { eprintln!("priority: I/O priority set to best-effort class 0 (highest)"); }
    } else {
        let e = std::io::Error::last_os_error();
        if debug { eprintln!("priority: ioprio_set failed ({})", e); }
    }
}

// ── Signal masking ──────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn mask_render_signals(debug: bool) {
    // Block signals that cause unnecessary context switches in the render thread.
    // SIGINT/SIGTERM are left unblocked so the process can still be killed.
    // Other threads inherit the mask but can unblock if needed.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGUSR1);
        libc::sigaddset(&mut set, libc::SIGUSR2);
        libc::sigaddset(&mut set, libc::SIGWINCH);
        libc::sigaddset(&mut set, libc::SIGURG);
        libc::sigaddset(&mut set, libc::SIGPWR);
        let ret = libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        if ret == 0 {
            if debug { eprintln!("priority: blocked SIGUSR1/2, SIGWINCH, SIGURG, SIGPWR on render thread"); }
        } else {
            if debug { eprintln!("priority: pthread_sigmask failed ({})", std::io::Error::from_raw_os_error(ret)); }
        }
    }
}

#[cfg(target_os = "macos")]
fn mask_render_signals(debug: bool) {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGUSR1);
        libc::sigaddset(&mut set, libc::SIGUSR2);
        libc::sigaddset(&mut set, libc::SIGWINCH);
        let ret = libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        if ret == 0 {
            if debug { eprintln!("priority: blocked SIGUSR1/2, SIGWINCH on render thread"); }
        } else {
            if debug { eprintln!("priority: pthread_sigmask failed ({})", std::io::Error::from_raw_os_error(ret)); }
        }
    }
}

// ── Huge pages advisory ─────────────────────────────────────────────────

/// Advise the kernel to back a memory region with transparent huge pages.
/// Call on mmap'd capture buffers for fewer TLB misses.
/// Best-effort — silently does nothing if THP is unavailable.
#[cfg(target_os = "linux")]
pub fn advise_hugepages(ptr: *mut u8, len: usize) {
    const MADV_HUGEPAGE: libc::c_int = 14;
    unsafe { libc::madvise(ptr as *mut libc::c_void, len, MADV_HUGEPAGE); }
}

#[cfg(not(target_os = "linux"))]
pub fn advise_hugepages(_ptr: *mut u8, _len: usize) {}

// ── D-Bus idle/screensaver inhibit ──────────────────────────────────────

/// Inhibit screen saver and idle timeout via D-Bus.
/// Returns a cookie that must be kept alive (dropping it un-inhibits).
/// Falls back through: portal → freedesktop screensaver → KWin compositing.
pub fn inhibit_idle(debug: bool) -> Option<IdleInhibit> {
    // Try org.freedesktop.portal.Inhibit first (works on Flatpak, Snap, native)
    if let Some(i) = try_portal_inhibit(debug) { return Some(i); }
    // Try org.freedesktop.ScreenSaver.Inhibit (KDE, XFCE, MATE)
    if let Some(i) = try_screensaver_inhibit(debug) { return Some(i); }
    if debug { eprintln!("priority: no idle inhibit method available"); }
    None
}

/// Try KWin compositing suspend (separate from idle inhibit).
/// Returns a handle that resumes compositing on drop.
#[cfg(target_os = "linux")]
pub fn try_suspend_compositing(debug: bool) -> Option<CompositingSuspend> {
    // org.kde.KWin /Compositor org.kde.kwin.Compositing.suspend
    let output = std::process::Command::new("dbus-send")
        .args([
            "--session", "--print-reply", "--type=method_call",
            "--dest=org.kde.KWin",
            "/Compositor",
            "org.kde.kwin.Compositing.suspend",
        ])
        .output()
        .ok()?;
    if output.status.success() {
        if debug { eprintln!("priority: KWin compositing suspended"); }
        Some(CompositingSuspend { _private: () })
    } else {
        if debug { eprintln!("priority: KWin compositing suspend not available"); }
        None
    }
}

#[cfg(not(target_os = "linux"))]
pub fn try_suspend_compositing(_debug: bool) -> Option<CompositingSuspend> { None }

pub struct CompositingSuspend { _private: () }

impl Drop for CompositingSuspend {
    fn drop(&mut self) {
        let _ = std::process::Command::new("dbus-send")
            .args([
                "--session", "--type=method_call",
                "--dest=org.kde.KWin",
                "/Compositor",
                "org.kde.kwin.Compositing.resume",
            ])
            .output();
    }
}

pub struct IdleInhibit {
    kind: InhibitKind,
}

enum InhibitKind {
    Portal,
    ScreenSaver(u32),
}

fn try_portal_inhibit(debug: bool) -> Option<IdleInhibit> {
    // org.freedesktop.portal.Inhibit.Inhibit(window_id, flags, options)
    // flags=8 means idle inhibit
    let output = std::process::Command::new("dbus-send")
        .args([
            "--session", "--print-reply", "--type=method_call",
            "--dest=org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Inhibit.Inhibit",
            "string:", "uint32:8",
            "dict:string:variant:",
        ])
        .output()
        .ok()?;
    if output.status.success() {
        if debug { eprintln!("priority: idle inhibited via portal"); }
        Some(IdleInhibit { kind: InhibitKind::Portal })
    } else {
        None
    }
}

fn try_screensaver_inhibit(debug: bool) -> Option<IdleInhibit> {
    let output = std::process::Command::new("dbus-send")
        .args([
            "--session", "--print-reply", "--type=method_call",
            "--dest=org.freedesktop.ScreenSaver",
            "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.Inhibit",
            "string:capview",
            "string:capture card viewer - low latency",
        ])
        .output()
        .ok()?;
    if output.status.success() {
        // Parse cookie from reply: "   uint32 <cookie>\n"
        let stdout = String::from_utf8_lossy(&output.stdout);
        let cookie = stdout.split_whitespace()
            .rev()
            .find_map(|w| w.parse::<u32>().ok())
            .unwrap_or(0);
        if debug { eprintln!("priority: screensaver inhibited (cookie={})", cookie); }
        Some(IdleInhibit { kind: InhibitKind::ScreenSaver(cookie) })
    } else {
        None
    }
}

impl Drop for IdleInhibit {
    fn drop(&mut self) {
        match &self.kind {
            InhibitKind::Portal => {
                // Portal inhibit is per-connection; dropping the bus connection releases it.
                // Nothing to do explicitly.
            }
            InhibitKind::ScreenSaver(cookie) => {
                let _ = std::process::Command::new("dbus-send")
                    .args([
                        "--session", "--type=method_call",
                        "--dest=org.freedesktop.ScreenSaver",
                        "/org/freedesktop/ScreenSaver",
                        "org.freedesktop.ScreenSaver.UnInhibit",
                        &format!("uint32:{}", cookie),
                    ])
                    .output();
            }
        }
    }
}

fn main() {
    // Build timestamp for the version stamp shown at the bottom of the OSD
    // menu. Cargo reruns build.rs when any crate source changes, so the
    // stamp refreshes on every real rebuild. Honours SOURCE_DATE_EPOCH
    // for reproducible/Docker builds.
    let now_secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        });
    let stamp = format_utc_stamp(now_secs);
    println!("cargo:rustc-env=CAPVIEW_BUILD_TIME={}", stamp);

    #[cfg(target_os = "linux")]
    {
        // Link PulseAudio libraries for audio passthrough
        println!("cargo:rustc-link-lib=pulse");
        println!("cargo:rustc-link-lib=pulse-simple");
    }

    #[cfg(target_os = "macos")]
    {
        // macOS frameworks for AVFoundation capture, CoreAudio, AppKit clipboard
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=CoreAudio");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=dylib=objc");
    }
}

/// Convert a Unix timestamp (UTC) to "YYYYMMDD-HHMMSS". Uses Howard
/// Hinnant's civil_from_days algorithm (public domain); avoids pulling
/// in chrono/time as a build dep.
fn format_utc_stamp(secs: u64) -> String {
    let day = (secs / 86_400) as i64;
    let sec_of_day = (secs % 86_400) as u32;
    let hour = sec_of_day / 3600;
    let minute = (sec_of_day % 3600) / 60;
    let second = sec_of_day % 60;

    // days since 1970-01-01 -> civil (year, month, day)
    let z = day + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + if m <= 2 { 1 } else { 0 };

    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        year, m, d, hour, minute, second
    )
}

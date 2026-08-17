//! On-screen display with embedded 8×8 bitmap font.
//!
//! Displays brief messages (brightness, mute, screenshot) in the
//! bottom-left corner of the window, mplayer-style.  Uses a glyph
//! atlas texture built at init time — zero external dependencies.

use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, Canvas, Texture, TextureCreator};
use sdl2::video::{Window, WindowContext};
use std::time::{Duration, Instant};

pub const GLYPH_W: u32 = 8;
pub const GLYPH_H: u32 = 8;
pub const FIRST_CHAR: u8 = 32;
pub const LAST_CHAR: u8 = 126;
const NUM_GLYPHS: usize = (LAST_CHAR - FIRST_CHAR + 1) as usize; // 95
pub const ATLAS_W: u32 = NUM_GLYPHS as u32 * GLYPH_W; // 760
pub const ATLAS_H: u32 = GLYPH_H; // 8

const OSD_BG_ALPHA_DEFAULT: u8 = 160;

// ── Embedded 8×8 bitmap font (ASCII 32-126) ─────────────────────────
//
// Each glyph is 8 bytes, one per scanline, MSB = leftmost pixel.
// Designed for clean readability at 2–5× scaling.

const FONT: [[u8; 8]; NUM_GLYPHS] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // 32 ' '
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00], // 33 '!'
    [0x6C, 0x6C, 0x6C, 0x00, 0x00, 0x00, 0x00, 0x00], // 34 '"'
    [0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00], // 35 '#'
    [0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00], // 36 '$'
    [0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00], // 37 '%'
    [0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00], // 38 '&'
    [0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00], // 39 '''
    [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00], // 40 '('
    [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00], // 41 ')'
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // 42 '*'
    [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00], // 43 '+'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30], // 44 ','
    [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00], // 45 '-'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00], // 46 '.'
    [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00], // 47 '/'
    [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00], // 48 '0'
    [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00], // 49 '1'
    [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00], // 50 '2'
    [0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00], // 51 '3'
    [0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00], // 52 '4'
    [0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00], // 53 '5'
    [0x3C, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00], // 54 '6'
    [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00], // 55 '7'
    [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00], // 56 '8'
    [0x3C, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x3C, 0x00], // 57 '9'
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00], // 58 ':'
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00], // 59 ';'
    [0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00], // 60 '<'
    [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00], // 61 '='
    [0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x00], // 62 '>'
    [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00], // 63 '?'
    [0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00], // 64 '@'
    [0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00], // 65 'A'
    [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00], // 66 'B'
    [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00], // 67 'C'
    [0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00], // 68 'D'
    [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00], // 69 'E'
    [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00], // 70 'F'
    [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00], // 71 'G'
    [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00], // 72 'H'
    [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00], // 73 'I'
    [0x3E, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3C, 0x00], // 74 'J'
    [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00], // 75 'K'
    [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00], // 76 'L'
    [0xC6, 0xEE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6, 0x00], // 77 'M'
    [0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00], // 78 'N'
    [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00], // 79 'O'
    [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00], // 80 'P'
    [0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00], // 81 'Q'
    [0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00], // 82 'R'
    [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00], // 83 'S'
    [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // 84 'T'
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00], // 85 'U'
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00], // 86 'V'
    [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00], // 87 'W'
    [0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00], // 88 'X'
    [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00], // 89 'Y'
    [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00], // 90 'Z'
    [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00], // 91 '['
    [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00], // 92 '\'
    [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00], // 93 ']'
    [0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00], // 94 '^'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00], // 95 '_'
    [0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00], // 96 '`'
    [0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00], // 97 'a'
    [0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00], // 98 'b'
    [0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00], // 99 'c'
    [0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00], // 100 'd'
    [0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00], // 101 'e'
    [0x1C, 0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x00], // 102 'f'
    [0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C], // 103 'g'
    [0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00], // 104 'h'
    [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00], // 105 'i'
    [0x0C, 0x00, 0x1C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38], // 106 'j'
    [0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00], // 107 'k'
    [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00], // 108 'l'
    [0x00, 0x00, 0xCC, 0xFE, 0xD6, 0xC6, 0xC6, 0x00], // 109 'm'
    [0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00], // 110 'n'
    [0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00], // 111 'o'
    [0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60], // 112 'p'
    [0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06], // 113 'q'
    [0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00], // 114 'r'
    [0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00], // 115 's'
    [0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x1C, 0x00], // 116 't'
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00], // 117 'u'
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00], // 118 'v'
    [0x00, 0x00, 0xC6, 0xC6, 0xD6, 0xFE, 0x6C, 0x00], // 119 'w'
    [0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00], // 120 'x'
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C], // 121 'y'
    [0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00], // 122 'z'
    [0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00], // 123 '{'
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // 124 '|'
    [0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00], // 125 '}'
    [0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // 126 '~'
];

// ── OSD slot system ─────────────────────────────────────────────────
//
// Slots are rendered top-to-bottom in priority order.  Each slot is
// either empty, persistent (pinned), or timed (auto-expires).
// Active slots stack vertically in the top-right corner.

/// Display slots, ordered by render priority (top → bottom).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Slot {
    Fps = 0,       // always-on FPS counter (highest, topmost)
    Status = 1,    // persistent indicators like "Recording"
    Streaming = 2, // persistent streaming state (server/client)
    Strip = 3,     // analysis strip frame counter
    Transient = 4, // timed messages: brightness, mute, saved, …
}

const NUM_SLOTS: usize = 5;

struct SlotState {
    text: String,
    expires: Option<Instant>, // None = pinned (persistent), Some = timed
}

impl SlotState {
    fn empty() -> Self {
        Self { text: String::new(), expires: Some(Instant::now()) }
    }

    fn active(&self) -> bool {
        if self.text.is_empty() { return false; }
        match self.expires {
            None => true,           // pinned
            Some(t) => Instant::now() < t,
        }
    }
}

pub struct Osd<'a> {
    atlas: Option<Texture<'a>>,
    slots: [SlotState; NUM_SLOTS],
    // Centre menu (tree with submenu navigation)
    menu_open: bool,
    menu_root: Vec<MenuItem>,
    menu_path: Vec<usize>,
    menu_cursor_stack: Vec<usize>,
    menu_cursor: usize,
    // Text field editing state
    text_editing: bool,
    text_cursor: usize,
    // Help overlay (left side)
    help_visible: bool,
    // Dynamic extra help lines (appended after static lines)
    extra_help: Vec<String>,
    // Dirty flag: set when OSD content changes, cleared by take_dirty()
    dirty: bool,
    // Track which slots were active last tick (to detect expiry transitions)
    was_active: [bool; NUM_SLOTS],
    // Background alpha (0–255)
    bg_alpha: u8,
    // Separate dirty flag for VK renderer (not consumed by take_dirty)
    vk_dirty: bool,
    // Bottom-right indicator text (e.g. "Paused")
    bottom_right: Option<String>,
}

// ── Centre menu item ────────────────────────────────────────────────

pub enum MenuItem {
    /// A value-cycling item (left/right to change).
    Value {
        label: String,
        values: Vec<String>,
        selected: usize,
    },
    /// A submenu container (Enter / Right to open).
    SubMenu {
        label: String,
        children: Vec<MenuItem>,
    },
    /// A numeric entry item (left/right to adjust, wraps at bounds).
    #[allow(dead_code)]
    Numeric {
        label: String,
        value: i32,
        min: i32,
        max: i32,
    },
    /// A free-text entry field (Enter to edit, type to fill, Enter to confirm).
    Text {
        label: String,
        value: String,
    },
    /// A visual separator (blank row, not selectable).
    Separator,
    /// Non-selectable informational text (e.g. version stamp).
    Info {
        text: String,
    },
}

impl MenuItem {
    /// Create a value-cycling item.
    pub fn value(label: &str, values: &[&str], selected: usize) -> Self {
        MenuItem::Value {
            label: label.to_string(),
            values: values.iter().map(|s| s.to_string()).collect(),
            selected,
        }
    }
    /// Create a submenu container.
    pub fn submenu(label: &str, children: Vec<MenuItem>) -> Self {
        MenuItem::SubMenu {
            label: label.to_string(),
            children,
        }
    }
    /// Action item (no cycling values, no children).
    #[allow(dead_code)]
    pub fn action(label: &str) -> Self {
        MenuItem::Value {
            label: label.to_string(),
            values: Vec::new(),
            selected: 0,
        }
    }
    /// Create a numeric entry item with min/max bounds (wraps on overflow).
    #[allow(dead_code)]
    pub fn numeric(label: &str, value: i32, min: i32, max: i32) -> Self {
        MenuItem::Numeric {
            label: label.to_string(),
            value: value.clamp(min, max),
            min,
            max,
        }
    }
    /// Create a free-text entry field.
    pub fn text(label: &str, value: &str) -> Self {
        MenuItem::Text {
            label: label.to_string(),
            value: value.to_string(),
        }
    }
    /// Create a visual separator (blank row).
    pub fn separator() -> Self {
        MenuItem::Separator
    }
    /// Create a non-selectable informational row (e.g. version stamp).
    pub fn info(text: &str) -> Self {
        MenuItem::Info { text: text.to_string() }
    }
    /// The item's display label.
    #[allow(dead_code)]
    pub fn label(&self) -> &str {
        match self {
            MenuItem::Value { label, .. }
            | MenuItem::SubMenu { label, .. }
            | MenuItem::Numeric { label, .. }
            | MenuItem::Text { label, .. } => label,
            MenuItem::Info { text } => text,
            MenuItem::Separator => "",
        }
    }
}

// ── Help key list (static) ──────────────────────────────────────────

const HELP_LINES: &[&str] = &[
    "F1          Help",
    "F4          FPS (dev/cap/render)",
    "F5          Toggle stream server",
    "F9          Record video",
    "F12         Screenshot (clipboard)",
    "Shift+F12   Screenshot (save)",
    "S+Ctrl+F12  Screenshot (tar)",
    "Tab         Menu",
    "+/-         Brightness",
    "PgUp/Dn     Volume",
    "F           Fullscreen",
    "M           Mute audio",
    "P           Pause capture",
    "Q           Quit",
    "Esc         Back / Close",
];

/// Build the glyph atlas as an R8 pixel buffer for OpenGL upload.
pub fn build_gl_atlas() -> Vec<u8> {
    let w = ATLAS_W as usize;
    let h = ATLAS_H as usize;
    let mut pixels = vec![0u8; w * h];
    for (ci, glyph) in FONT.iter().enumerate() {
        for row in 0..8usize {
            for col in 0..8usize {
                let on = (glyph[row] >> (7 - col)) & 1 != 0;
                let x = ci * 8 + col;
                let idx = row * w + x;
                pixels[idx] = if on { 255 } else { 0 };
            }
        }
    }
    pixels
}

/// Convert ARGB u32 (0xAARRGGBB) to GL float colour.
fn argb_to_gl(argb: u32) -> [f32; 4] {
    let a = ((argb >> 24) & 0xFF) as f32 / 255.0;
    let r = ((argb >> 16) & 0xFF) as f32 / 255.0;
    let g = ((argb >> 8) & 0xFF) as f32 / 255.0;
    let b = (argb & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

// ── Menu tree navigation helpers ────────────────────────────────────

fn navigate_items<'a>(root: &'a [MenuItem], path: &[usize]) -> &'a [MenuItem] {
    if path.is_empty() { return root; }
    match &root[path[0]] {
        MenuItem::SubMenu { children, .. } => navigate_items(children, &path[1..]),
        _ => root,
    }
}

fn navigate_items_mut<'a>(root: &'a mut [MenuItem], path: &[usize]) -> &'a mut [MenuItem] {
    if path.is_empty() { return root; }
    match &mut root[path[0]] {
        MenuItem::SubMenu { children, .. } => navigate_items_mut(children, &path[1..]),
        _ => unreachable!(),
    }
}

fn menu_title(root: &[MenuItem], path: &[usize]) -> String {
    if path.is_empty() { return "Settings".to_string(); }
    let mut items = root;
    let mut title = "Settings";
    for &idx in path {
        match &items[idx] {
            MenuItem::SubMenu { label, children, .. } => {
                title = label;
                items = children;
            }
            _ => break,
        }
    }
    title.to_string()
}

impl<'a> Osd<'a> {
    /// Build the glyph atlas and return a ready-to-use OSD.
    pub fn new(tc: &'a TextureCreator<WindowContext>) -> anyhow::Result<Self> {
        let mut atlas = tc
            .create_texture_streaming(PixelFormatEnum::ARGB8888, ATLAS_W, ATLAS_H)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        atlas.set_blend_mode(BlendMode::Blend);

        // Nearest-neighbor: crisp pixel-art look when scaled
        unsafe {
            sdl2_sys::SDL_SetTextureScaleMode(
                atlas.raw(),
                sdl2_sys::SDL_ScaleMode::SDL_ScaleModeNearest,
            );
        }

        // Rasterise every glyph into the atlas as white-on-transparent
        atlas
            .with_lock(None, |pixels, pitch| {
                for (ci, glyph) in FONT.iter().enumerate() {
                    for row in 0..8usize {
                        for col in 0..8usize {
                            let on = (glyph[row] >> (7 - col)) & 1 != 0;
                            let x = ci * 8 + col;
                            let off = row * pitch + x * 4;
                            let px: u32 = if on { 0xFFFF_FFFF } else { 0x0000_0000 };
                            pixels[off..off + 4].copy_from_slice(&px.to_ne_bytes());
                        }
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok(Self {
            atlas: Some(atlas),
            slots: [SlotState::empty(), SlotState::empty(), SlotState::empty(), SlotState::empty(), SlotState::empty()],
            menu_open: false,
            menu_root: Vec::new(),
            menu_path: Vec::new(),
            menu_cursor_stack: Vec::new(),
            menu_cursor: 0,
            text_editing: false,
            text_cursor: 0,
            help_visible: false,
            extra_help: Vec::new(),
            dirty: false,
            was_active: [false; NUM_SLOTS],
            bg_alpha: OSD_BG_ALPHA_DEFAULT,
            vk_dirty: false,
            bottom_right: None,
        })
    }

    /// Create an OSD without an SDL texture atlas (for Vulkan mode).
    /// Menu/slot data model works, but SDL render() will be a no-op.
    pub fn new_headless() -> anyhow::Result<Self> {
        Ok(Self {
            atlas: None,
            slots: [SlotState::empty(), SlotState::empty(), SlotState::empty(), SlotState::empty(), SlotState::empty()],
            menu_open: false,
            menu_root: Vec::new(),
            menu_path: Vec::new(),
            menu_cursor_stack: Vec::new(),
            menu_cursor: 0,
            text_editing: false,
            text_cursor: 0,
            help_visible: false,
            extra_help: Vec::new(),
            dirty: false,
            was_active: [false; NUM_SLOTS],
            bg_alpha: OSD_BG_ALPHA_DEFAULT,
            vk_dirty: false,
            bottom_right: None,
        })
    }

    // ── public API ──────────────────────────────────────────────────

    /// Set or clear the bottom-right indicator text.
    pub fn set_bottom_right(&mut self, text: Option<String>) {
        self.bottom_right = text;
        self.dirty = true; self.vk_dirty = true;
    }

    /// Set background opacity from a 0–100 percentage.
    pub fn set_opacity(&mut self, pct: u32) {
        self.bg_alpha = (pct.min(100) as f32 * 2.55) as u8;
        self.dirty = true; self.vk_dirty = true;
    }

    /// Show a timed message in a slot (auto-expires after `duration_ms`).
    pub fn show(&mut self, slot: Slot, msg: impl Into<String>, duration_ms: u64) {
        let s = &mut self.slots[slot as usize];
        s.text = msg.into();
        s.expires = Some(Instant::now() + Duration::from_millis(duration_ms));
        self.dirty = true; self.vk_dirty = true;
    }

    /// Pin a persistent message in a slot (stays until cleared).
    pub fn pin(&mut self, slot: Slot, msg: impl Into<String>) {
        let s = &mut self.slots[slot as usize];
        s.text = msg.into();
        s.expires = None; // persistent
        self.dirty = true; self.vk_dirty = true;
    }

    /// Clear a slot (remove its message immediately).
    pub fn clear(&mut self, slot: Slot) {
        let s = &mut self.slots[slot as usize];
        s.text.clear();
        s.expires = Some(Instant::now());
        self.dirty = true; self.vk_dirty = true;
    }

    /// True if any OSD element is currently visible.
    #[allow(dead_code)]
    pub fn visible(&self) -> bool {
        self.slots.iter().any(|s| s.active()) || self.menu_open || self.help_visible || self.bottom_right.is_some()
    }

    /// Check for timed-slot expiry transitions and return+clear the dirty flag.
    /// Call once per main-loop iteration, before checking `dirty`.
    pub fn take_dirty(&mut self) -> bool {
        // Detect slots that just expired (were active last tick, not anymore)
        for i in 0..NUM_SLOTS {
            let now_active = self.slots[i].active();
            if self.was_active[i] && !now_active {
                self.dirty = true; self.vk_dirty = true;
            }
            self.was_active[i] = now_active;
        }
        let d = self.dirty;
        self.dirty = false;
        d
    }

    // ── Menu API ────────────────────────────────────────────────────

    pub fn set_menu_items(&mut self, items: Vec<MenuItem>) {
        self.menu_root = items;
        self.menu_path.clear();
        self.menu_cursor_stack.clear();
        self.menu_cursor = 0;
    }

    pub fn menu_open(&self) -> bool { self.menu_open }

    pub fn toggle_menu(&mut self) {
        self.text_editing = false;
        self.menu_open = !self.menu_open;
        self.dirty = true; self.vk_dirty = true;
        if self.menu_open {
            self.help_visible = false;
            // Always re-open at root level
            self.menu_path.clear();
            self.menu_cursor_stack.clear();
            self.menu_cursor = 0;
        }
    }

    /// Go back one submenu level, or close the menu if at root.
    pub fn menu_back(&mut self) {
        if !self.menu_open { return; }
        self.text_editing = false;
        self.dirty = true; self.vk_dirty = true;
        if self.menu_path.is_empty() {
            self.menu_open = false;
        } else {
            self.menu_path.pop();
            self.menu_cursor = self.menu_cursor_stack.pop().unwrap_or(0);
        }
    }

    pub fn menu_up(&mut self) {
        if !self.menu_open { return; }
        self.text_editing = false;
        let items = navigate_items(&self.menu_root, &self.menu_path);
        let mut c = self.menu_cursor;
        while c > 0 {
            c -= 1;
            if !matches!(items.get(c), Some(MenuItem::Separator) | Some(MenuItem::Info { .. })) {
                self.menu_cursor = c;
                self.dirty = true; self.vk_dirty = true;
                return;
            }
        }
    }

    pub fn menu_down(&mut self) {
        if !self.menu_open { return; }
        self.text_editing = false;
        let items = navigate_items(&self.menu_root, &self.menu_path);
        let len = items.len();
        let mut c = self.menu_cursor;
        while c + 1 < len {
            c += 1;
            if !matches!(items.get(c), Some(MenuItem::Separator) | Some(MenuItem::Info { .. })) {
                self.menu_cursor = c;
                self.dirty = true; self.vk_dirty = true;
                return;
            }
        }
    }

    #[allow(dead_code)]
    pub fn menu_left(&mut self) {
        if !self.menu_open { return; }
        self.menu_adjust(-1);
    }

    #[allow(dead_code)]
    pub fn menu_right(&mut self) {
        self.menu_right_by(1);
    }

    /// Right arrow with variable step (Shift+Right for ×10 etc.).
    pub fn menu_right_by(&mut self, step: i32) {
        if !self.menu_open { return; }
        self.dirty = true; self.vk_dirty = true;
        let item_kind = {
            let items = navigate_items(&self.menu_root, &self.menu_path);
            match items.get(self.menu_cursor) {
                Some(MenuItem::SubMenu { .. }) => 0,
                Some(MenuItem::Text { .. }) => 1,
                _ => 2,
            }
        };
        match item_kind {
            0 => {
                // Enter submenu
                self.menu_cursor_stack.push(self.menu_cursor);
                self.menu_path.push(self.menu_cursor);
                self.menu_cursor = 0;
            }
            1 => {
                // Start editing text field
                self.start_editing_text();
            }
            _ => {
                self.menu_adjust(step);
            }
        }
    }

    /// Adjust the current item's value by `delta`.
    /// Value items clamp; Numeric items wrap around.
    pub fn menu_adjust(&mut self, delta: i32) {
        self.dirty = true; self.vk_dirty = true;
        let cursor = self.menu_cursor;
        let items = navigate_items_mut(&mut self.menu_root, &self.menu_path);
        match items.get_mut(cursor) {
            Some(MenuItem::Value { values, selected, .. }) => {
                if !values.is_empty() {
                    let len = values.len();
                    if delta < 0 {
                        *selected = selected.saturating_sub((-delta) as usize);
                    } else {
                        *selected = (*selected + delta as usize).min(len - 1);
                    }
                }
            }
            Some(MenuItem::Numeric { value, min, max, .. }) => {
                let range = (*max - *min + 1) as i64;
                let new_val = *value as i64 + delta as i64;
                *value = *min + (((new_val - *min as i64) % range + range) % range) as i32;
            }
            _ => {}
        }
    }

    /// Enter a submenu (if the cursor is on one). Returns true if entered.
    pub fn menu_enter(&mut self) -> bool {
        if !self.menu_open { return false; }
        self.dirty = true; self.vk_dirty = true;
        let is_submenu = {
            let items = navigate_items(&self.menu_root, &self.menu_path);
            matches!(items.get(self.menu_cursor), Some(MenuItem::SubMenu { .. }))
        };
        if is_submenu {
            self.menu_cursor_stack.push(self.menu_cursor);
            self.menu_path.push(self.menu_cursor);
            self.menu_cursor = 0;
            true
        } else {
            false
        }
    }

    /// Find a Value item by label anywhere in the menu tree.
    /// Returns (selected_index, num_values).
    pub fn find_menu_value(&self, label: &str) -> Option<(usize, usize)> {
        fn search(items: &[MenuItem], label: &str) -> Option<(usize, usize)> {
            for item in items {
                match item {
                    MenuItem::Value { label: l, selected, values } if l == label => {
                        return Some((*selected, values.len()));
                    }
                    MenuItem::SubMenu { children, .. } => {
                        if let Some(r) = search(children, label) { return Some(r); }
                    }
                    _ => {}
                }
            }
            None
        }
        search(&self.menu_root, label)
    }

    /// Set the selected index of a named Value menu item.
    pub fn set_menu_value(&mut self, label: &str, index: usize) {
        fn set_in(items: &mut [MenuItem], label: &str, index: usize) -> bool {
            for item in items {
                match item {
                    MenuItem::Value { label: l, selected, values } if l == label => {
                        *selected = index.min(values.len().saturating_sub(1));
                        return true;
                    }
                    MenuItem::SubMenu { children, .. } => {
                        if set_in(children, label, index) { return true; }
                    }
                    _ => {}
                }
            }
            false
        }
        set_in(&mut self.menu_root, label, index);
    }

    /// Replace the children of a named root-level submenu.
    pub fn set_submenu_items(&mut self, submenu_label: &str, new_children: Vec<MenuItem>) {
        // Search root level first, then one level deep (for nested submenus)
        for item in self.menu_root.iter_mut() {
            if let MenuItem::SubMenu { label, children } = item {
                if label == submenu_label {
                    *children = new_children;
                    return;
                }
                for child in children.iter_mut() {
                    if let MenuItem::SubMenu { label: ref cl, children: ref mut cc } = child {
                        if cl == submenu_label {
                            *cc = new_children;
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Returns the label of the current action item (Value with no values).
    pub fn menu_action_label(&self) -> Option<String> {
        let items = navigate_items(&self.menu_root, &self.menu_path);
        match items.get(self.menu_cursor) {
            Some(MenuItem::Value { label, values, .. }) if values.is_empty() => {
                Some(label.clone())
            }
            _ => None,
        }
    }

    /// Change the label of an action item (Value with no values) found
    /// recursively by its current label.
    pub fn set_action_label(&mut self, old_label: &str, new_label: &str) {
        fn update(items: &mut [MenuItem], old: &str, new: &str) -> bool {
            for item in items.iter_mut() {
                match item {
                    MenuItem::Value { label, values, .. }
                        if values.is_empty() && label == old =>
                    {
                        *label = new.to_string();
                        return true;
                    }
                    MenuItem::SubMenu { children, .. } => {
                        if update(children, old, new) { return true; }
                    }
                    _ => {}
                }
            }
            false
        }
        update(&mut self.menu_root, old_label, new_label);
    }

    /// Find a Numeric item by label anywhere in the menu tree.
    #[allow(dead_code)]
    pub fn find_menu_numeric(&self, label: &str) -> Option<i32> {
        fn search(items: &[MenuItem], label: &str) -> Option<i32> {
            for item in items {
                match item {
                    MenuItem::Numeric { label: l, value, .. } if l == label => {
                        return Some(*value);
                    }
                    MenuItem::SubMenu { children, .. } => {
                        if let Some(r) = search(children, label) { return Some(r); }
                    }
                    _ => {}
                }
            }
            None
        }
        search(&self.menu_root, label)
    }

    /// Find a Text item by label anywhere in the menu tree.
    pub fn find_menu_text(&self, label: &str) -> Option<String> {
        fn search(items: &[MenuItem], label: &str) -> Option<String> {
            for item in items {
                match item {
                    MenuItem::Text { label: l, value } if l == label => {
                        return Some(value.clone());
                    }
                    MenuItem::SubMenu { children, .. } => {
                        if let Some(r) = search(children, label) { return Some(r); }
                    }
                    _ => {}
                }
            }
            None
        }
        search(&self.menu_root, label)
    }

    // ── Text editing API ────────────────────────────────────────────

    /// True if a text field is currently being edited.
    pub fn is_editing_text(&self) -> bool { self.text_editing }

    /// Begin editing the text field at the current cursor position.
    /// Returns true if the current item is a Text field.
    pub fn start_editing_text(&mut self) -> bool {
        let cursor = self.menu_cursor;
        let items = navigate_items(&self.menu_root, &self.menu_path);
        if let Some(MenuItem::Text { value, .. }) = items.get(cursor) {
            self.text_editing = true;
            self.text_cursor = value.len();
            true
        } else {
            false
        }
    }

    /// Stop editing. Returns the final text value if we were editing, else None.
    pub fn stop_editing_text(&mut self) -> Option<String> {
        if !self.text_editing { return None; }
        self.text_editing = false;
        let cursor = self.menu_cursor;
        let items = navigate_items(&self.menu_root, &self.menu_path);
        if let Some(MenuItem::Text { value, .. }) = items.get(cursor) {
            Some(value.clone())
        } else {
            None
        }
    }

    /// Cancel editing and restore original value is not needed here —
    /// we just stop editing; the value stays as-is.
    pub fn cancel_editing_text(&mut self) {
        self.text_editing = false;
    }

    /// Insert a character at the text cursor position.
    pub fn text_insert(&mut self, ch: char) {
        if !self.text_editing { return; }
        self.dirty = true; self.vk_dirty = true;
        let cursor = self.menu_cursor;
        let items = navigate_items_mut(&mut self.menu_root, &self.menu_path);
        if let Some(MenuItem::Text { value, .. }) = items.get_mut(cursor) {
            if self.text_cursor > value.len() {
                self.text_cursor = value.len();
            }
            value.insert(self.text_cursor, ch);
            self.text_cursor += 1;
        }
    }

    /// Delete the character before the text cursor (backspace).
    pub fn text_backspace(&mut self) {
        if !self.text_editing { return; }
        self.dirty = true; self.vk_dirty = true;
        let cursor = self.menu_cursor;
        let items = navigate_items_mut(&mut self.menu_root, &self.menu_path);
        if let Some(MenuItem::Text { value, .. }) = items.get_mut(cursor) {
            if self.text_cursor > 0 && self.text_cursor <= value.len() {
                value.remove(self.text_cursor - 1);
                self.text_cursor -= 1;
            }
        }
    }

    /// Move text cursor left.
    pub fn text_cursor_left(&mut self) {
        if self.text_cursor > 0 { self.text_cursor -= 1; }
    }

    /// Move text cursor right.
    pub fn text_cursor_right(&mut self) {
        if !self.text_editing { return; }
        let cursor = self.menu_cursor;
        let items = navigate_items(&self.menu_root, &self.menu_path);
        if let Some(MenuItem::Text { value, .. }) = items.get(cursor) {
            if self.text_cursor < value.len() {
                self.text_cursor += 1;
            }
        }
    }

    /// Get the text cursor position (for rendering).
    #[allow(dead_code)]
    pub fn text_cursor_pos(&self) -> usize { self.text_cursor }

    // ── Help API ────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn help_visible(&self) -> bool { self.help_visible }

    pub fn toggle_help(&mut self) {
        self.help_visible = !self.help_visible;
        self.dirty = true; self.vk_dirty = true;
        if self.help_visible { self.menu_open = false; }
    }

    /// Set extra help lines (appended after static key bindings).
    pub fn set_extra_help(&mut self, lines: Vec<String>) {
        self.extra_help = lines;
    }

    /// Collect all help lines (static + dynamic extra).
    fn all_help_lines(&self) -> Vec<&str> {
        let mut lines: Vec<&str> = HELP_LINES.to_vec();
        for l in &self.extra_help {
            lines.push(l.as_str());
        }
        lines
    }

    // ── Rendering ───────────────────────────────────────────────────

    /// Render all active slots stacked top-right, highest priority on top.
    pub fn render(&self, canvas: &mut Canvas<Window>, win_w: u32, win_h: u32) {
        self.render_slots(canvas, win_w, win_h);
        if self.help_visible {
            self.render_help(canvas, win_w, win_h);
        }
        if self.menu_open {
            self.render_menu(canvas, win_w, win_h);
        }
        self.render_bottom_right_sdl(canvas, win_w, win_h);
    }

    fn render_slots(&self, canvas: &mut Canvas<Window>, win_w: u32, win_h: u32) {
        let any = self.slots.iter().any(|s| s.active());
        if !any { return; }

        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;

        let mut y_cursor = (pad * 2) as i32;

        for slot in &self.slots {
            if !slot.active() { continue; }
            let lines: Vec<&str> = slot.text.split('\n').collect();
            let max_w = lines.iter().map(|l| l.len() as u32 * gw).max().unwrap_or(0);
            let block_h = lines.len() as u32 * row_h + (lines.len().saturating_sub(1) as u32) * pad;
            let x0 = (win_w - max_w - pad * 2) as i32;

            canvas.set_blend_mode(BlendMode::Blend);
            canvas.set_draw_color(sdl2::pixels::Color::RGBA(0, 0, 0, self.bg_alpha));
            let _ = canvas.fill_rect(Rect::new(
                x0 - pad as i32,
                y_cursor - pad as i32,
                max_w + pad * 2,
                block_h + pad * 2,
            ));

            for line in &lines {
                self.draw_text(canvas, line, x0, y_cursor, scale);
                y_cursor += row_h as i32 + pad as i32;
            }
        }
    }

    fn render_bottom_right_sdl(&self, canvas: &mut Canvas<Window>, win_w: u32, win_h: u32) {
        let text = match &self.bottom_right {
            Some(t) => t.as_str(),
            None => return,
        };
        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let tw = text.len() as u32 * gw;
        let x0 = (win_w - tw - pad * 2) as i32;
        let y0 = (win_h - gh - pad * 4) as i32;
        canvas.set_blend_mode(BlendMode::Blend);
        canvas.set_draw_color(sdl2::pixels::Color::RGBA(0, 0, 0, self.bg_alpha));
        let _ = canvas.fill_rect(Rect::new(x0 - pad as i32, y0 - pad as i32,
            tw + pad * 2, gh + pad * 2));
        self.draw_text(canvas, text, x0, y0, scale);
    }

    fn render_help(&self, canvas: &mut Canvas<Window>, _win_w: u32, win_h: u32) {
        let lines = self.all_help_lines();
        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let line_spacing = row_h + pad;
        let num_lines = lines.len() as u32;

        // Find widest line for background box
        let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u32;
        let box_w = max_chars * gw + pad * 4;
        let box_h = num_lines * line_spacing + pad * 2;

        let bx = (pad * 2) as i32;
        let by = (pad * 2) as i32;

        // Background
        canvas.set_blend_mode(BlendMode::Blend);
        canvas.set_draw_color(sdl2::pixels::Color::RGBA(0, 0, 0, self.bg_alpha));
        let _ = canvas.fill_rect(Rect::new(bx, by, box_w, box_h));

        // Title
        let title = "Key Bindings";
        let tx = bx + pad as i32;
        let mut ty = by + pad as i32;
        self.draw_text_color(canvas, title, tx, ty, scale, 0xFF_FF_FF_00); // yellow
        ty += line_spacing as i32;

        // Lines
        for line in &lines {
            self.draw_text(canvas, line, tx, ty, scale);
            ty += line_spacing as i32;
        }
    }

    fn render_menu(&self, canvas: &mut Canvas<Window>, win_w: u32, win_h: u32) {
        let items = navigate_items(&self.menu_root, &self.menu_path);
        if items.is_empty() && self.menu_path.is_empty() { return; }

        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let line_spacing = row_h + pad;

        let title = menu_title(&self.menu_root, &self.menu_path);

        // Build display strings
        let rows: Vec<String> = items.iter().enumerate().map(|(i, item)| {
            match item {
                MenuItem::Value { label, values, selected } => {
                    if values.is_empty() {
                        format!("[ {} ]", label)
                    } else {
                        let val = &values[*selected];
                        let left = if *selected > 0 { "<" } else { " " };
                        let right = if *selected + 1 < values.len() { ">" } else { " " };
                        format!("{}: {} {} {}", label, left, val, right)
                    }
                }
                MenuItem::Numeric { label, value, .. } => {
                    format!("{}: < {} >", label, value)
                }
                MenuItem::Text { label, value } => {
                    if self.text_editing && i == self.menu_cursor {
                        // Show cursor as pipe at insertion point
                        let pos = self.text_cursor.min(value.len());
                        let (before, after) = value.split_at(pos);
                        format!("{}: {}|{}", label, before, after)
                    } else {
                        format!("{}: {}", label, value)
                    }
                }
                MenuItem::SubMenu { label, .. } => format!("{}  >", label),
                MenuItem::Separator => String::new(),
                MenuItem::Info { text } => text.clone(),
            }
        }).collect();

        let max_chars = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u32;
        let box_w = max_chars * gw + pad * 4;
        let title_w = title.len() as u32;
        let box_w = box_w.max(title_w * gw + pad * 4);
        let num_rows = rows.len() as u32 + 1; // +1 for title
        let box_h = num_rows * line_spacing + pad * 2;

        // Centre
        let bx = ((win_w - box_w) / 2) as i32;
        let by = ((win_h - box_h) / 2) as i32;

        // Semi-transparent background
        canvas.set_blend_mode(BlendMode::Blend);
        canvas.set_draw_color(sdl2::pixels::Color::RGBA(0, 0, 0, 200));
        let _ = canvas.fill_rect(Rect::new(bx, by, box_w, box_h));

        let tx = bx + pad as i32 * 2;
        let mut ty = by + pad as i32;

        // Title
        self.draw_text_color(canvas, &title, tx, ty, scale, 0xFF_FF_FF_00);
        ty += line_spacing as i32;

        // Items
        for (i, row_text) in rows.iter().enumerate() {
            if i == self.menu_cursor {
                // Highlight bar
                canvas.set_draw_color(sdl2::pixels::Color::RGBA(255, 255, 255, 40));
                let _ = canvas.fill_rect(Rect::new(
                    bx + pad as i32,
                    ty - pad as i32,
                    box_w - pad * 2,
                    row_h,
                ));
                self.draw_text_color(canvas, row_text, tx, ty, scale, 0xFF_FF_FF_FF);
            } else {
                self.draw_text_color(canvas, row_text, tx, ty, scale, 0xFF_C0_C0_C0);
            }
            ty += line_spacing as i32;
        }
    }

    // ── Text primitives ─────────────────────────────────────────────

    fn draw_text(&self, canvas: &mut Canvas<Window>, text: &str, x: i32, y: i32, scale: u32) {
        self.draw_text_color(canvas, text, x, y, scale, 0xFF_FF_FF_FF);
    }

    fn draw_text_color(&self, canvas: &mut Canvas<Window>, text: &str, x: i32, y: i32, scale: u32, argb: u32) {
        let atlas = match self.atlas {
            Some(ref a) => a,
            None => return,
        };
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let r = ((argb >> 16) & 0xFF) as u8;
        let g = ((argb >> 8) & 0xFF) as u8;
        let b = (argb & 0xFF) as u8;
        // Tint atlas via raw SDL (avoids &mut self requirement on Texture)
        unsafe { sdl2_sys::SDL_SetTextureColorMod(atlas.raw(), r, g, b); }
        for (i, ch) in text.chars().enumerate() {
            let code = ch as u32;
            if code < FIRST_CHAR as u32 || code > LAST_CHAR as u32 { continue; }
            let ci = (code - FIRST_CHAR as u32) as i32;
            let src = Rect::new(ci * GLYPH_W as i32, 0, GLYPH_W, GLYPH_H);
            let dst = Rect::new(x + (i as u32 * gw) as i32, y, gw, gh);
            let _ = canvas.copy(atlas, Some(src), Some(dst));
        }
        // Reset to white
        unsafe { sdl2_sys::SDL_SetTextureColorMod(atlas.raw(), 255, 255, 255); }
    }

    // ── GL rendering (native OpenGL OSD — no SDL canvas) ────────────

    /// Render all OSD elements via native GL (used in OpenGL renderer path).
    pub fn render_gl(&self, gl: &crate::gl_renderer::GlRenderer, win_w: u32, win_h: u32) {
        let has_content = self.slots.iter().any(|s| s.active())
            || self.help_visible
            || self.menu_open
            || self.bottom_right.is_some();
        if !has_content { return; }

        gl.begin_osd(win_w, win_h);
        self.render_slots_gl(gl, win_w, win_h);
        if self.help_visible {
            self.render_help_gl(gl, win_w, win_h);
        }
        if self.menu_open {
            self.render_menu_gl(gl, win_w, win_h);
        }
        self.render_bottom_right_gl(gl, win_w, win_h);
        gl.end_osd();
    }

    fn render_slots_gl(&self, gl: &crate::gl_renderer::GlRenderer, win_w: u32, win_h: u32) {
        let any = self.slots.iter().any(|s| s.active());
        if !any { return; }

        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;

        let mut y_cursor = (pad * 2) as i32;

        for slot in &self.slots {
            if !slot.active() { continue; }
            let lines: Vec<&str> = slot.text.split('\n').collect();
            let max_w = lines.iter().map(|l| l.len() as u32 * gw).max().unwrap_or(0);
            let block_h = lines.len() as u32 * row_h + (lines.len().saturating_sub(1) as u32) * pad;
            let x0 = (win_w - max_w - pad * 2) as i32;

            gl.osd_rect(
                x0 - pad as i32, y_cursor - pad as i32,
                max_w + pad * 2, block_h + pad * 2,
                [0.0, 0.0, 0.0, self.bg_alpha as f32 / 255.0],
                win_w, win_h,
            );

            for line in &lines {
                gl.osd_text(line, x0, y_cursor, scale,
                    [1.0, 1.0, 1.0, 1.0], win_w, win_h);
                y_cursor += row_h as i32 + pad as i32;
            }
        }
    }

    fn render_bottom_right_gl(&self, gl: &crate::gl_renderer::GlRenderer, win_w: u32, win_h: u32) {
        let text = match &self.bottom_right {
            Some(t) => t.as_str(),
            None => return,
        };
        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let tw = text.len() as u32 * gw;
        let x0 = (win_w - tw - pad * 2) as i32;
        let y0 = (win_h - gh - pad * 4) as i32;
        gl.osd_rect(x0 - pad as i32, y0 - pad as i32,
            tw + pad * 2, gh + pad * 2,
            [0.0, 0.0, 0.0, self.bg_alpha as f32 / 255.0], win_w, win_h);
        gl.osd_text(text, x0, y0, scale,
            [1.0, 1.0, 1.0, 1.0], win_w, win_h);
    }

    fn render_help_gl(&self, gl: &crate::gl_renderer::GlRenderer, win_w: u32, win_h: u32) {
        let lines = self.all_help_lines();
        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let line_spacing = row_h + pad;
        let num_lines = lines.len() as u32;

        let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u32;
        let box_w = max_chars * gw + pad * 4;
        let box_h = num_lines * line_spacing + pad * 2;

        let bx = (pad * 2) as i32;
        let by = (pad * 2) as i32;

        gl.osd_rect(bx, by, box_w, box_h,
            [0.0, 0.0, 0.0, self.bg_alpha as f32 / 255.0],
            win_w, win_h);

        let title = "Key Bindings";
        let tx = bx + pad as i32;
        let mut ty = by + pad as i32;
        gl.osd_text(title, tx, ty, scale,
            argb_to_gl(0xFF_FF_FF_00), win_w, win_h);
        ty += line_spacing as i32;

        for line in &lines {
            gl.osd_text(line, tx, ty, scale,
                [1.0, 1.0, 1.0, 1.0], win_w, win_h);
            ty += line_spacing as i32;
        }
    }

    fn render_menu_gl(&self, gl: &crate::gl_renderer::GlRenderer, win_w: u32, win_h: u32) {
        let items = navigate_items(&self.menu_root, &self.menu_path);
        if items.is_empty() && self.menu_path.is_empty() { return; }

        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let line_spacing = row_h + pad;

        let title = menu_title(&self.menu_root, &self.menu_path);

        let rows: Vec<String> = items.iter().enumerate().map(|(i, item)| {
            match item {
                MenuItem::Value { label, values, selected } => {
                    if values.is_empty() {
                        format!("[ {} ]", label)
                    } else {
                        let val = &values[*selected];
                        let left = if *selected > 0 { "<" } else { " " };
                        let right = if *selected + 1 < values.len() { ">" } else { " " };
                        format!("{}: {} {} {}", label, left, val, right)
                    }
                }
                MenuItem::Numeric { label, value, .. } => {
                    format!("{}: < {} >", label, value)
                }
                MenuItem::Text { label, value } => {
                    if self.text_editing && i == self.menu_cursor {
                        let pos = self.text_cursor.min(value.len());
                        let (before, after) = value.split_at(pos);
                        format!("{}: {}|{}", label, before, after)
                    } else {
                        format!("{}: {}", label, value)
                    }
                }
                MenuItem::SubMenu { label, .. } => format!("{}  >", label),
                MenuItem::Separator => String::new(),
                MenuItem::Info { text } => text.clone(),
            }
        }).collect();

        let max_chars = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u32;
        let box_w = max_chars * gw + pad * 4;
        let title_w = title.len() as u32;
        let box_w = box_w.max(title_w * gw + pad * 4);
        let num_rows = rows.len() as u32 + 1;
        let box_h = num_rows * line_spacing + pad * 2;

        let bx = ((win_w - box_w) / 2) as i32;
        let by = ((win_h - box_h) / 2) as i32;

        gl.osd_rect(bx, by, box_w, box_h,
            [0.0, 0.0, 0.0, 200.0 / 255.0],
            win_w, win_h);

        let tx = bx + pad as i32 * 2;
        let mut ty = by + pad as i32;

        gl.osd_text(&title, tx, ty, scale,
            argb_to_gl(0xFF_FF_FF_00), win_w, win_h);
        ty += line_spacing as i32;

        for (i, row_text) in rows.iter().enumerate() {
            if i == self.menu_cursor {
                gl.osd_rect(
                    bx + pad as i32, ty - pad as i32,
                    box_w - pad * 2, row_h,
                    [1.0, 1.0, 1.0, 40.0 / 255.0],
                    win_w, win_h,
                );
                gl.osd_text(row_text, tx, ty, scale,
                    [1.0, 1.0, 1.0, 1.0], win_w, win_h);
            } else {
                gl.osd_text(row_text, tx, ty, scale,
                    argb_to_gl(0xFF_C0_C0_C0), win_w, win_h);
            }
            ty += line_spacing as i32;
        }
    }

    // ── Vulkan rendering (software-rasterized OSD via VkRenderer) ───

    /// Render all OSD elements via VkRenderer's software rasterizer.
    /// Skips re-rendering when content hasn't changed (VK renderer keeps
    /// the previous OSD texture on the GPU for compositing).
    pub fn render_vk(&mut self, vk: &mut crate::vk_renderer::VkRenderer, win_w: u32, win_h: u32) {
        let has_content = self.slots.iter().any(|s| s.active())
            || self.help_visible
            || self.menu_open
            || self.bottom_right.is_some();
        if !has_content {
            if vk.osd_has_content() {
                vk.osd_clear();
            }
            return;
        }

        // Only re-rasterize OSD when content actually changed
        if !self.vk_dirty {
            vk.mark_osd_content();
            return;
        }
        self.vk_dirty = false;

        vk.begin_osd(win_w, win_h);
        self.render_slots_vk(vk, win_w, win_h);
        if self.help_visible {
            self.render_help_vk(vk, win_w, win_h);
        }
        if self.menu_open {
            self.render_menu_vk(vk, win_w, win_h);
        }
        self.render_bottom_right_vk(vk, win_w, win_h);
        vk.end_osd();
    }

    fn render_slots_vk(&self, vk: &mut crate::vk_renderer::VkRenderer, win_w: u32, win_h: u32) {
        let any = self.slots.iter().any(|s| s.active());
        if !any { return; }

        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let mut y_cursor = (pad * 2) as i32;

        for slot in &self.slots {
            if !slot.active() { continue; }
            let lines: Vec<&str> = slot.text.split('\n').collect();
            let max_w = lines.iter().map(|l| l.len() as u32 * gw).max().unwrap_or(0);
            let block_h = lines.len() as u32 * row_h + (lines.len().saturating_sub(1) as u32) * pad;
            let x0 = (win_w - max_w - pad * 2) as i32;
            vk.osd_rect(x0 - pad as i32, y_cursor - pad as i32,
                max_w + pad * 2, block_h + pad * 2,
                [0.0, 0.0, 0.0, self.bg_alpha as f32 / 255.0], win_w, win_h);
            for line in &lines {
                vk.osd_text(line, x0, y_cursor, scale,
                    [1.0, 1.0, 1.0, 1.0], win_w, win_h);
                y_cursor += row_h as i32 + pad as i32;
            }
        }
    }

    fn render_bottom_right_vk(&self, vk: &mut crate::vk_renderer::VkRenderer, win_w: u32, win_h: u32) {
        let text = match &self.bottom_right {
            Some(t) => t.as_str(),
            None => return,
        };
        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let tw = text.len() as u32 * gw;
        let x0 = (win_w - tw - pad * 2) as i32;
        let y0 = (win_h - gh - pad * 4) as i32;
        vk.osd_rect(x0 - pad as i32, y0 - pad as i32,
            tw + pad * 2, gh + pad * 2,
            [0.0, 0.0, 0.0, self.bg_alpha as f32 / 255.0], win_w, win_h);
        vk.osd_text(text, x0, y0, scale,
            [1.0, 1.0, 1.0, 1.0], win_w, win_h);
    }

    fn render_help_vk(&self, vk: &mut crate::vk_renderer::VkRenderer, win_w: u32, win_h: u32) {
        let lines = self.all_help_lines();
        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let line_spacing = row_h + pad;
        let num_lines = lines.len() as u32;

        let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u32;
        let box_w = max_chars * gw + pad * 4;
        let box_h = num_lines * line_spacing + pad * 2;

        let bx = (pad * 2) as i32;
        let by = (pad * 2) as i32;
        vk.osd_rect(bx, by, box_w, box_h,
            [0.0, 0.0, 0.0, self.bg_alpha as f32 / 255.0], win_w, win_h);

        let title = "Key Bindings";
        let tx = bx + pad as i32;
        let mut ty = by + pad as i32;
        vk.osd_text(title, tx, ty, scale,
            argb_to_gl(0xFF_FF_FF_00), win_w, win_h);
        ty += line_spacing as i32;
        for line in &lines {
            vk.osd_text(line, tx, ty, scale,
                [1.0, 1.0, 1.0, 1.0], win_w, win_h);
            ty += line_spacing as i32;
        }
    }

    fn render_menu_vk(&self, vk: &mut crate::vk_renderer::VkRenderer, win_w: u32, win_h: u32) {
        let items = navigate_items(&self.menu_root, &self.menu_path);
        if items.is_empty() && self.menu_path.is_empty() { return; }

        let scale = (win_h / 360).max(1).min(5);
        let gw = GLYPH_W * scale;
        let gh = GLYPH_H * scale;
        let pad = scale * 3;
        let row_h = gh + pad * 2;
        let line_spacing = row_h + pad;

        let title = menu_title(&self.menu_root, &self.menu_path);
        let rows: Vec<String> = items.iter().enumerate().map(|(i, item)| {
            match item {
                MenuItem::Value { label, values, selected } => {
                    if values.is_empty() {
                        format!("[ {} ]", label)
                    } else {
                        let val = &values[*selected];
                        let left = if *selected > 0 { "<" } else { " " };
                        let right = if *selected + 1 < values.len() { ">" } else { " " };
                        format!("{}: {} {} {}", label, left, val, right)
                    }
                }
                MenuItem::Numeric { label, value, .. } => {
                    format!("{}: < {} >", label, value)
                }
                MenuItem::Text { label, value } => {
                    if self.text_editing && i == self.menu_cursor {
                        let pos = self.text_cursor.min(value.len());
                        let (before, after) = value.split_at(pos);
                        format!("{}: {}|{}", label, before, after)
                    } else {
                        format!("{}: {}", label, value)
                    }
                }
                MenuItem::SubMenu { label, .. } => format!("{}  >", label),
                MenuItem::Separator => String::new(),
                MenuItem::Info { text } => text.clone(),
            }
        }).collect();

        let max_chars = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u32;
        let box_w = max_chars * gw + pad * 4;
        let title_w = title.len() as u32;
        let box_w = box_w.max(title_w * gw + pad * 4);
        let num_rows = rows.len() as u32 + 1;
        let box_h = num_rows * line_spacing + pad * 2;

        let bx = ((win_w - box_w) / 2) as i32;
        let by = ((win_h - box_h) / 2) as i32;
        vk.osd_rect(bx, by, box_w, box_h,
            [0.0, 0.0, 0.0, 200.0 / 255.0], win_w, win_h);

        let tx = bx + pad as i32 * 2;
        let mut ty = by + pad as i32;
        vk.osd_text(&title, tx, ty, scale,
            argb_to_gl(0xFF_FF_FF_00), win_w, win_h);
        ty += line_spacing as i32;

        for (i, row_text) in rows.iter().enumerate() {
            if i == self.menu_cursor {
                vk.osd_rect(bx + pad as i32, ty - pad as i32,
                    box_w - pad * 2, row_h,
                    [1.0, 1.0, 1.0, 40.0 / 255.0], win_w, win_h);
                vk.osd_text(row_text, tx, ty, scale,
                    [1.0, 1.0, 1.0, 1.0], win_w, win_h);
            } else {
                vk.osd_text(row_text, tx, ty, scale,
                    argb_to_gl(0xFF_C0_C0_C0), win_w, win_h);
            }
            ty += line_spacing as i32;
        }
    }
}

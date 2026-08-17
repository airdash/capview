.PHONY: build release install clean docker docker-run macos

build:
	cargo build

release:
	cargo build --release

# macOS native build (uses system frameworks, Homebrew deps)
# SDL2 from Homebrew may not be symlinked; set LIBRARY_PATH so the linker finds it.
macos:
	LIBRARY_PATH="$$(brew --prefix sdl2 2>/dev/null)/lib:$$LIBRARY_PATH" cargo build --release

install: release
	install -Dm755 target/release/capview $(DESTDIR)/usr/local/bin/capview
	install -Dm644 capview.conf.example $(DESTDIR)/etc/capview/capview.conf.example

clean:
	cargo clean

# Docker-based build (Linux only): produces a single binary at OUTPUT_DIR (default ./build-output)
docker:
	./build.sh $(OUTPUT_DIR)

docker-run:
	docker run --rm -it \
		--device=/dev/video0 \
		-e WAYLAND_DISPLAY \
		-e XDG_RUNTIME_DIR \
		-v $$XDG_RUNTIME_DIR/$$WAYLAND_DISPLAY:$$XDG_RUNTIME_DIR/$$WAYLAND_DISPLAY \
		capview

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path


def _insert_paths() -> None:
    vendor_dir = os.environ.get("HAG_VENDOR_DIR", "")
    for candidate in (vendor_dir, str(Path(__file__).resolve().parents[1])):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)


_insert_paths()

import webview

if sys.platform == "darwin":
    try:
        import AppKit
        import Foundation
        from PyObjCTools import AppHelper
        from webview.platforms import cocoa as cocoa_platform
    except Exception:
        AppKit = None
        Foundation = None
        AppHelper = None
        cocoa_platform = None
else:
    AppKit = None
    Foundation = None
    AppHelper = None
    cocoa_platform = None

from tools.webui import CUSTOM_CSS, build_demo, theme


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _raise_window_on_launch(window) -> None:
    if sys.platform not in ("win32", "darwin"):
        return

    try:
        if sys.platform == "darwin":
            if not (AppKit and Foundation and AppHelper and cocoa_platform):
                return

            def _focus_native_window() -> None:
                try:
                    app = AppKit.NSApplication.sharedApplication()
                    if hasattr(app, "setActivationPolicy_"):
                        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
                    native = cocoa_platform.BrowserView.instances.get(window.uid)
                    if native is None:
                        return
                    try:
                        native.window.setLevel_(AppKit.NSStatusWindowLevel)
                    except Exception:
                        pass
                    try:
                        native.window.orderFrontRegardless()
                    except Exception:
                        pass
                    try:
                        native.window.makeKeyAndOrderFront_(native.window)
                    except Exception:
                        pass
                    try:
                        app.activateIgnoringOtherApps_(Foundation.YES)
                    except Exception:
                        pass
                except Exception as error:
                    print(f"Hallway Avatar Gen webview: mac focus error: {error}")

            def _normalize_window_level() -> None:
                try:
                    native = cocoa_platform.BrowserView.instances.get(window.uid)
                    if native is not None:
                        native.window.setLevel_(AppKit.NSNormalWindowLevel)
                except Exception:
                    pass

            def _focus_burst() -> None:
                for _index in range(8):
                    try:
                        AppHelper.callAfter(_focus_native_window)
                    except Exception:
                        break
                    time.sleep(0.18)
                try:
                    AppHelper.callAfter(_normalize_window_level)
                except Exception:
                    pass

            threading.Thread(target=_focus_burst, daemon=True).start()
            return

        window.on_top = True
        time.sleep(0.8)
        window.on_top = False
    except Exception as error:
        print(f"Hallway Avatar Gen webview: raise window error: {error}")


def main() -> int:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    def _launch_gradio() -> None:
        demo = build_demo()
        demo.queue()
        demo.launch(
            inbrowser=False,
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            css=CUSTOM_CSS,
            theme=theme,
        )

    server_thread = threading.Thread(target=_launch_gradio, daemon=True)
    server_thread.start()

    time.sleep(2.0)
    window = webview.create_window("Hallway Avatar Gen", url, width=1480, height=980)
    try:
        window.shown += lambda: _raise_window_on_launch(window)
    except Exception:
        pass
    webview.start(debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

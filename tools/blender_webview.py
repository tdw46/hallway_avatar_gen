from __future__ import annotations

import os
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

from tools.webui import APP_HTML, HallwayWebApp


def _inject_js(window, js_code: str) -> None:
    try:
        window.evaluate_js(js_code)
    except Exception as error:
        print(f"Hallway Avatar Gen webview: JS inject error: {error}", flush=True)


def _inject_console_bridge(window) -> None:
    bridge_js = r"""
(function() {
  if (window.__hagConsoleBridge) return;
  window.__hagConsoleBridge = true;
  const send = (level, args) => {
    try {
      if (window.pywebview && window.pywebview.api && window.pywebview.api.on_console) {
        window.pywebview.api.on_console({level: level, args: args});
      }
    } catch (e) {}
  };
  ['log', 'warn', 'error', 'info', 'debug'].forEach(level => {
    const orig = console[level];
    console[level] = function(...args) {
      try { send(level, args); } catch (e) {}
      try { orig.apply(console, args); } catch (e) {}
    };
  });
  window.addEventListener('error', function(e) {
    try {
      const target = e.target || e.srcElement;
      if (target && target !== window) {
        const src = target.currentSrc || target.src || target.href || target.baseURI || '';
        send('error', ['resource-error', target.tagName || 'unknown', src]);
        return;
      }
      send('error', [e.message || 'error', e.filename || '', e.lineno || 0, e.colno || 0]);
    } catch (ex) {}
  }, true);
  window.addEventListener('unhandledrejection', function(e) {
    try {
      send('error', ['unhandledrejection', e.reason && (e.reason.message || e.reason) || e.reason]);
    } catch (ex) {}
  });
})();
"""
    _inject_js(window, bridge_js)


def _start_console_bridge(window) -> None:
    _inject_console_bridge(window)


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
                    print(f"Hallway Avatar Gen webview: mac focus error: {error}", flush=True)

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
        print(f"Hallway Avatar Gen webview: raise window error: {error}", flush=True)


def main() -> int:
    print("Hallway Avatar Gen webview: helper booting", flush=True)
    js_api = HallwayWebApp()

    def _on_loaded(window) -> None:
        print("Hallway Avatar Gen webview: window loaded", flush=True)
        _start_console_bridge(window)
        try:
            window.evaluate_js("window.dispatchEvent(new Event('pywebviewready'));")
        except Exception as error:
            print(f"Hallway Avatar Gen webview: pywebviewready dispatch error: {error}", flush=True)

    window = webview.create_window(
        "Hallway Avatar Gen",
        html=APP_HTML,
        width=1480,
        height=980,
        js_api=js_api,
        text_select=True,
    )
    js_api.bind_window(window)
    try:
        window.shown += lambda: _raise_window_on_launch(window)
    except Exception:
        pass
    def _on_closed(*_args) -> None:
        print("Hallway Avatar Gen webview: window closed", flush=True)
        js_api.shutdown()

    for candidate in (
        getattr(getattr(window, "events", None), "closed", None),
        getattr(window, "closed", None),
    ):
        if candidate is None:
            continue
        try:
            candidate += _on_closed
            break
        except Exception:
            continue

    try:
        webview.start(_on_loaded, (window,), debug=False)
    finally:
        js_api.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

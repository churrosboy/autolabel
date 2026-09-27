"""Tiny review UI (standard library only).

GET  /                      -> index.html
GET  /api/labels            -> [{clip_id,index,keyframe_type,coc,filter,review,sheet}]
GET  /api/stats
POST /api/review            {clip_id,index,review,coc?}
GET  /frames/<clip>/kfNNN/sheet.jpg
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..pipeline import Project

HERE = Path(__file__).parent


def make_handler(project: Project):
    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                body = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/labels":
                rows = []
                for l in project.labels():
                    sheet = project.frames_dir / l.clip_id / f"kf{l.index:03d}" / "sheet.jpg"
                    d = l.to_dict()
                    d["sheet"] = f"/frames/{l.clip_id}/kf{l.index:03d}/sheet.jpg" if sheet.exists() else None
                    d.pop("raw", None)
                    rows.append(d)
                self._json(rows)
            elif u.path == "/api/stats":
                self._json(project.stats())
            elif u.path.startswith("/frames/"):
                p = (project.frames_dir / u.path[len("/frames/"):]).resolve()
                if project.frames_dir.resolve() in p.parents and p.exists():
                    body = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/review":
                lab = project.set_review(data["clip_id"], int(data["index"]), data.get("review", "approved"), data.get("coc"))
                self._json({"ok": lab is not None, "label": lab.to_dict() if lab else None})
            elif u.path == "/api/export":
                self._json(project.export(data.get("name", "dataset"), reviewed_only=bool(data.get("reviewed_only"))))
            else:
                self.send_error(404)

    return H


def serve(project: Project, port: int = 8765, open_browser: bool = True):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(project))
    url = f"http://127.0.0.1:{port}/"
    print(f"review UI: {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

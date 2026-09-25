"""Kleiner lokaler HTTP-Server für die Live-Animation.

    python scripts/serve_animation.py            # dann http://127.0.0.1:8765 öffnen

`/` liefert web/hvac-agent-animation.html, `/state.json` den aktuellen Zustand aus
runs/live/anim_state.json (geschrieben von logic/watch.py::record(), z. B. im Reiter
"Beobachten"). Port 8765, weil 8000 schon von BOPTEST belegt ist. Nur an 127.0.0.1 gebunden.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / 'web' / 'hvac-agent-animation.html'
STATE = ROOT / 'runs' / 'live' / 'anim_state.json'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html', '/hvac-agent-animation.html'):
            self._send(PAGE, 'text/html; charset=utf-8')
        elif path == '/state.json':
            if STATE.exists():
                self._send(STATE, 'application/json')
            else:   # noch keine Episode gelaufen — die Seite zeigt dann "Warte auf Daten"
                self.send_error(404, 'runs/live/anim_state.json existiert noch nicht')
        else:
            self.send_error(404)

    def _send(self, file: Path, content_type: str):
        try:
            body = file.read_bytes()
        except OSError:     # gerade durch os.replace ersetzt — der nächste Abruf klappt
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):   # nicht jede Abfrage (alle 1,5 s) ins Terminal schreiben
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Live-Animation: http://127.0.0.1:{args.port}  (Strg+C beendet)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

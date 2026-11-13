"""Local read-only monitoring; serves only the dashboard and public progress JSON."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse
import webbrowser


def make_server(folder, port=8767):
    progress = Path(folder).resolve()/'progress.json'
    html = (Path(__file__).parent/'monitor.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route == '/':
                body, mime = html, 'text/html; charset=utf-8'
            elif route == '/api/status':
                try: body = progress.read_bytes()
                except FileNotFoundError:
                    body = json.dumps({'status':'waiting_for_training', 'history':[]}).encode()
                mime = 'application/json'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve(folder, port=8767, open_browser=False):
    server = make_server(folder, port)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Monitoring {Path(folder).resolve()/"progress.json"}\n{url}', flush=True)
    if open_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='system_runs/demo')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    serve(args.output, args.port, args.open)

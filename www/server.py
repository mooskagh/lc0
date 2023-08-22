import http.server
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

PORT = 8999

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

Handler.extensions_map = {
  '.html': 'text/html',
  '.png': 'image/png',
  '.jpg': 'image/jpg',
  '.wasm': 'application/wasm',
  '.css': 'text/css',
  '.js': 'application/x-javascript',
  '': 'application/octet-stream',
}

httpd = socketserver.TCPServer(("", PORT), Handler)

print("serving at port", PORT)
httpd.serve_forever()

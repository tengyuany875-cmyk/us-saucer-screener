// Serve only the three generated preview pages on this computer.
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');

const root = path.join(__dirname, 'output', 'local-ui');
const port = Number(process.argv[2] || 8767);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid preview port');
const allowed = new Set(['index.html', 'daily.html', 'recent.html']);
http.createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, {Allow: 'GET, HEAD'}).end();
    return;
  }
  const name = req.url === '/' ? 'daily.html' : req.url.slice(1);
  if (!allowed.has(name)) {
    res.writeHead(404).end('Not found');
    return;
  }
  try {
    const body = await fs.readFile(path.join(root, name));
    res.writeHead(200, {'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff', 'Content-Length': body.length});
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch (error) {
    res.writeHead(503).end('Build the local preview with pullback_ui.py first.');
  }
}).listen(port, '127.0.0.1', () => console.log(`Local preview: http://127.0.0.1:${port}/daily.html`));

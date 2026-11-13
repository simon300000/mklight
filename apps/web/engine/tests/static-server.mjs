// A plain static file server for the built harness (dist/), like GitHub
// Pages: no isolation headers unless `--isolate` (then COOP/COEP are sent,
// like a host that can set headers; `--isolate-200` sends them on 200s only
// and answers revalidations with a bare 304, like many servers, which WebKit
// then refuses for worker scripts: the engine must cope).
// Usage: node static-server.mjs <dir> <port> [--isolate | --isolate-200]
import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { extname, join, normalize } from 'node:path'

const [dir, port, flag] = process.argv.slice(2)
const isolate = flag === '--isolate' || flag === '--isolate-200'
const bare304 = flag === '--isolate-200'
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.css': 'text/css',
}

createServer(async (req, res) => {
  let path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '')
  let file = join(dir, path)
  try {
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html')
    const body = await readFile(file)
    const etag = `"${body.length}-${(await stat(file)).mtimeMs}"`
    if (bare304 && req.headers['if-none-match'] === etag) {
      res.writeHead(304, { ETag: etag })
      res.end()
      return
    }
    const headers = {
      'Content-Type': types[extname(file)] ?? 'application/octet-stream',
      'Cache-Control': bare304 ? 'no-cache' : 'no-store',
    }
    if (bare304) headers.ETag = etag
    if (isolate) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin'
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    }
    res.writeHead(200, headers)
    res.end(body)
  } catch {
    res.writeHead(404)
    res.end('not found')
  }
}).listen(Number(port), '127.0.0.1')

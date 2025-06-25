// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'save-chat-history',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.method === 'POST' && req.url === '/__save_history') {
            let body = ''
            req.on('data', chunk => (body += chunk))
            req.on('end', () => {
              const file = path.resolve(__dirname, 'chat_history.json')
              let newEntry
              try {
                newEntry = JSON.parse(body)
              } catch {
                // Invalid JSON — just return success without writing
                res.writeHead(200, { 'Content-Type': 'application/json' })
                res.end(JSON.stringify({ status: 'ok' }))
                return
              }

              // Read existing entries
              let entries = []
              if (fs.existsSync(file)) {
                const data = fs.readFileSync(file, 'utf-8')
                entries = data
                  .split('\n')
                  .filter(line => line.trim())
                  .map(line => {
                    try {
                      return JSON.parse(line)
                    } catch {
                      return null
                    }
                  })
                  .filter(Boolean)
              }

              // Replace or append
              const idx = entries.findIndex(e => e.sessionId === newEntry.sessionId)
              if (idx !== -1) {
                entries[idx] = newEntry
              } else {
                entries.push(newEntry)
              }

              // Write back all entries
              const output = entries.map(e => JSON.stringify(e)).join('\n') + '\n'
              fs.writeFileSync(file, output, 'utf-8')

              res.writeHead(200, { 'Content-Type': 'application/json' })
              res.end(JSON.stringify({ status: 'ok' }))
            })
          } else {
            next()
          }
        })
      }
    }
  ],

  server: {
    open: true
  }
})

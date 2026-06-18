import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import path from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import { jsPDF } from 'jspdf'
import { chromium } from 'playwright'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '..')
const outputDir = path.resolve(projectRoot, 'output', 'pdf')
const outputFile = path.resolve(outputDir, 'edgeanomalycctv-framework.pdf')
const previewHost = '127.0.0.1'
const previewPort = 4173
const previewUrl = `http://${previewHost}:${previewPort}/`
const screenshotWidth = 1600
const screenshotHeight = 900

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function runCommand(command, args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: 'inherit',
    })

    child.on('exit', (code) => {
      if (code === 0) {
        resolve()
        return
      }

      reject(new Error(`${command} ${args.join(' ')} exited with code ${code ?? 'unknown'}`))
    })
  })
}

async function waitForServer(url, timeoutMs = 20000) {
  const startedAt = Date.now()

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok) return
    } catch {
      // Keep polling until the preview server is ready.
    }

    await sleep(250)
  }

  throw new Error(`Timed out waiting for preview server at ${url}`)
}

async function ensureSlideAssetsLoaded(page) {
  await page.waitForFunction(() => {
    return Array.from(document.querySelectorAll('.slide-panel img')).every((img) => img.complete)
  })
}

async function exportSlides() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'edgeanomaly-export-'))
  await runCommand('npm', ['run', 'build'], projectRoot)
  const previewProcess = spawn('npm', ['run', 'preview', '--', '--host', previewHost, '--port', String(previewPort)], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  let previewOutput = ''
  previewProcess.stdout.on('data', (chunk) => {
    previewOutput += chunk.toString()
  })
  previewProcess.stderr.on('data', (chunk) => {
    previewOutput += chunk.toString()
  })

  try {
    await waitForServer(previewUrl)
    await mkdir(outputDir, { recursive: true })

    const browser = await chromium.launch({ headless: true })

    try {
      const context = await browser.newContext({
        acceptDownloads: true,
        viewport: { width: 1720, height: 1080 },
        deviceScaleFactor: 2,
      })
      const page = await context.newPage()

      await page.goto(previewUrl, { waitUntil: 'networkidle' })
      await page.locator('.deck-shell').waitFor()
      await page.evaluate(() => {
        document.documentElement.classList.add('export-screenshot-mode')
        document.body.classList.add('export-screenshot-mode')
      })
      await page.waitForFunction(() => document.fonts?.status === 'loaded')

      const slidePanel = page.locator('.slide-panel').first()
      const totalSlides = await page.evaluate(() => {
        const visibleCounter = document.querySelector('.deck-shell')?.getAttribute('data-total-slides')
        if (visibleCounter) return Number(visibleCounter)

        const panels = document.querySelectorAll('section.slide-panel')
        return panels.length > 1 ? panels.length - 1 : 0
      })

      if (!Number.isFinite(totalSlides) || totalSlides < 1) {
        throw new Error('Could not determine total slide count from the app UI.')
      }

      const imagePaths = []

      for (let index = 0; index < totalSlides; index += 1) {
        await ensureSlideAssetsLoaded(page)
        await page.waitForTimeout(150)

        const imagePath = path.join(tempDir, `slide-${String(index + 1).padStart(2, '0')}.jpg`)
        await slidePanel.screenshot({
          path: imagePath,
          type: 'jpeg',
          quality: 84,
        })
        imagePaths.push(imagePath)

        if (index < totalSlides - 1) {
          await page.getByRole('button', { name: 'Next' }).click()
        }
      }

      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'pt',
        format: [screenshotWidth, screenshotHeight],
        compress: true,
      })

      for (const [index, imagePath] of imagePaths.entries()) {
        if (index > 0) {
          pdf.addPage([screenshotWidth, screenshotHeight], 'landscape')
        }

        const imageBuffer = await readFile(imagePath)
        const imageDataUrl = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`
        pdf.addImage(imageDataUrl, 'JPEG', 0, 0, screenshotWidth, screenshotHeight, undefined, 'MEDIUM')
      }

      pdf.save(outputFile)
      console.log(`Saved screenshot-based PDF to ${outputFile}`)
    } finally {
      await browser.close()
    }
  } catch (error) {
    console.error(previewOutput.trim())
    throw error
  } finally {
    previewProcess.kill('SIGINT')
    await rm(tempDir, { recursive: true, force: true })
  }
}

exportSlides().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})

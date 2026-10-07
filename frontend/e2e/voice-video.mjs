// End-to-end check of Aura's voice and camera features in a real browser.
// Chrome's fake devices play e2e/media/speech.wav as the microphone and e2e/media/face.mjpeg as the
// camera (generate them with `uv run python frontend/e2e/make_media.py`; they come from licensed
// datasets and are not committed).
//   node e2e/voice-video.mjs <screenshot-dir>      (expects Aura in full mode on http://localhost:8000)
import puppeteer from 'puppeteer-core'
import { fileURLToPath } from 'node:url'

const OUT = process.argv[2] ?? '.'
const BASE = process.env.URL ?? 'http://localhost:8000'
const media = (f) => fileURLToPath(new URL(`./media/${f}`, import.meta.url))
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${media('speech.wav')}`, `--use-file-for-fake-video-capture=${media('face.mjpeg')}`],
})
const page = await browser.newPage()
await page.setViewport({ width: 1366, height: 860 })
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
const wait = (ms) => new Promise((r) => setTimeout(r, ms))
const check = (name, ok, detail = '') => console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` - ${detail}` : ''}`)

await page.goto(`${BASE}/register`, { waitUntil: 'networkidle0' })
const inputs = await page.$$('form input')
await inputs[0].type('Rishi'); await inputs[1].type(`vv${Date.now()}@example.com`); await inputs[2].type('a-strong-password')
await page.click('button[type=submit]'); await page.waitForSelector('#composer')
await page.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => b.textContent.includes('Talk to Aura') && !b.disabled), { timeout: 30000 }).catch(() => {})
const startButtons = await page.$$eval('button', (bs) => bs.filter((b) => /Talk to Aura|Turn on camera/.test(b.textContent)).map((b) => !b.disabled))
check('start screen: Talk and Camera buttons are enabled', startButtons.length === 2 && startButtons.every(Boolean))
await page.screenshot({ path: `${OUT}/1-start.png` })

// camera: turning it on from the start screen opens a conversation and shows a live expression
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Turn on camera')).click())
await page.waitForFunction(() => location.pathname.startsWith('/chat/'), { timeout: 15000 })
const faceOk = await page.waitForFunction(() => document.querySelector('[aria-label="Your camera"]')?.innerText.match(/Anxiety|Sadness|Joy|Anger|Disgust|Surprise|Calm/), { timeout: 30000 }).then(() => true, () => false)
check('camera: live preview reads the face', faceOk, await page.$eval('[aria-label="Your camera"]', (el) => el.innerText.replace(/\n/g, ' | ')).catch(() => 'no preview'))
await page.screenshot({ path: `${OUT}/2-camera.png` })

// voice: record the speech clip, send it, and wait for transcript + reply
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Talk').click())
await wait(4500)
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Send').click())
// done = a stored user turn with its "Why this reply?" link and no reply still streaming
// (the camera's pulsing "Live" dot is outside the chat column, so it is ignored)
const replied = await page.waitForFunction(() => {
  const chat = document.querySelector('[aria-label="Conversation"]')
  return chat && chat.innerText.includes('Why this reply') && !chat.innerText.includes('Aura is thinking') && !chat.querySelector('.animate-pulse')
}, { timeout: 60000 }).then(() => true, () => false)
const bubble = await page.$$eval('div.bg-user-bubble', (els) => els.map((e) => e.innerText)).catch(() => [])
check('voice: speech transcribed into the chat', replied && bubble.some((t) => /door/i.test(t)), JSON.stringify(bubble))
const chips = await page.evaluate(() => document.body.innerText.match(/\bvoice\b|\bface\b/g) ?? [])
check('voice + face both blended into the reading', chips.includes('voice') && chips.includes('face'), chips.join(','))
await wait(800)
await page.screenshot({ path: `${OUT}/3-voice-reply.png` })
// phone layout: camera becomes a strip above the chat, nothing overflows sideways
await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true }) // reloads the page
await page.waitForSelector('#composer')
// after a reload the camera stays off (it never switches itself on); turn it on again
await page.click('button[aria-pressed="false"][title^="Let Aura see"]')
const stripOk = await page.waitForFunction(() => document.querySelector('[aria-label="Your camera"]')?.innerText.match(/Sadness|Anxiety|Joy|Anger|Disgust|Surprise|Calm/), { timeout: 30000 }).then(() => true, () => false)
check('phone: camera strip shows the expression', stripOk)
await wait(500)
await page.screenshot({ path: `${OUT}/4-mobile.png` })
const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
check('phone: no horizontal overflow', !overflow)
check('no page errors', errors.length === 0, errors.join(' | '))
await browser.close()

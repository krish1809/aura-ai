import { Brain, Ear, HeartHandshake, MessageCircleHeart, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { CrisisCard } from '../components/chat/CrisisCard'
import { useDialog } from '../components/Dialog'
import { Button, Card } from '../components/ui'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'
import type { CrisisResource, ModelReport, ModelStatus } from '../lib/types'

const STEPS = [
  {
    icon: Ear,
    title: 'Sense',
    body: 'Your words, your voice and (if you turn the camera on) your expression. Whisper transcribes speech; a wav2vec2 model reads emotion in your voice; a ResNet-18 trained on FER2013 reads your face.',
  },
  {
    icon: Brain,
    title: 'Understand',
    body: 'The signals are fused into one feeling. A RoBERTa model finds the cause in your own words, and smaller models pick the topic and the support strategy that fits this moment.',
  },
  {
    icon: MessageCircleHeart,
    title: 'Respond',
    body: 'Llama 3.2 3B, fine-tuned on real emotional-support conversations, writes the reply. It receives everything above in its instructions, which you can inspect under “Why this reply?”.',
  },
  {
    icon: HeartHandshake,
    title: 'Remember',
    body: 'People, causes and feelings you mention form a personal knowledge graph, and short notes carry context between conversations. You can see and erase both on the Memory page.',
  },
]

const COMPONENT_LABEL: Record<string, string> = {
  text_emotion: 'Emotion from text',
  ece: 'Cause extraction',
  ner: 'People & entities',
  topic: 'Topic',
  strategy: 'Support strategy',
  stt: 'Speech-to-text',
  ser: 'Emotion from voice',
  fer: 'Emotion from face',
  llm: 'Reply generation',
}

function fmt(m: ModelReport['metrics'][number]) {
  return m.kind === 'pct' ? `${(m.value * 100).toFixed(1)}%` : m.value.toFixed(2)
}

export function AboutPage() {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [report, setReport] = useState<ModelReport[] | null>(null)
  const [resources, setResources] = useState<CrisisResource[]>([])
  const { deleteAccount } = useAuth()
  const navigate = useNavigate()
  const dialog = useDialog()

  useEffect(() => {
    api.models().then(setStatus, () => setStatus(null))
    api.metrics().then((r) => setReport(r.models), () => setReport([]))
    api.resources().then((r) => setResources(r.crisis_resources), () => undefined)
  }, [])

  const removeAccount = async () => {
    const ok = await dialog.confirm({
      title: 'Delete your account?',
      body: 'Your account, conversations and everything Aura remembers will be permanently deleted. This cannot be undone.',
      confirmLabel: 'Delete my account',
      danger: true,
    })
    if (!ok) return
    await deleteAccount()
    navigate('/login')
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-10 px-4 py-6 sm:px-6">
        <header className="space-y-2">
          <h1 className="font-display text-3xl font-semibold tracking-tight">How Aura works</h1>
          <p className="max-w-2xl text-ink-2">
            Aura is a multimodal emotional-support companion. It doesn’t just read your words. It tries to understand how you
            feel and why, and it shows you its reasoning.
          </p>
        </header>

        <ol className="grid gap-3 sm:grid-cols-2">
          {STEPS.map(({ icon: Icon, title, body }, i) => (
            <li key={title}>
              <Card className="h-full space-y-2 p-4">
                <p className="flex items-center gap-2 font-semibold">
                  <span className="grid size-8 place-items-center rounded-lg bg-accent-soft text-accent">
                    <Icon size={16} aria-hidden />
                  </span>
                  {i + 1}. {title}
                </p>
                <p className="text-sm leading-relaxed text-ink-2">{body}</p>
              </Card>
            </li>
          ))}
        </ol>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Running right now</h2>
          {status ? (
            <Card className="divide-y divide-line">
              <p className="px-4 py-2.5 text-sm text-ink-2">
                Mode: <span className="font-medium text-ink">{status.mode === 'full' ? 'Full (trained models on GPU)' : 'Lite (no GPU: rules, small models, demo replies)'}</span>
              </p>
              {Object.entries(status.components.components).map(([name, c]) => (
                <div key={name} className="flex items-center justify-between gap-3 px-4 py-2 text-sm">
                  <span>{COMPONENT_LABEL[name] ?? name}</span>
                  <span className={c.ready ? 'text-ink-2' : 'text-ink-3'}>
                    <span className={`mr-1.5 inline-block size-2 rounded-full ${c.ready ? 'bg-[var(--emo-neutral)]' : 'bg-surface-3'}`} aria-hidden />
                    {c.ready ? c.source : 'not loaded'}
                  </span>
                </div>
              ))}
            </Card>
          ) : (
            <p className="text-sm text-ink-3">Model status unavailable.</p>
          )}
        </section>

        <section className="space-y-3">
          <div>
            <h2 className="text-lg font-semibold">Model report card</h2>
            <p className="text-sm text-ink-2">Measured on held-out test data that the models never saw in training.</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {report?.map((m) => (
              <Card key={m.id} className="space-y-2 p-4">
                <div>
                  <p className="font-medium">{m.title}</p>
                  <p className="text-xs text-ink-3">{m.protocol}</p>
                </div>
                {m.trained ? (
                  <dl className="space-y-1 text-sm">
                    {m.metrics.map((x) => (
                      <div key={x.label} className="flex justify-between gap-3">
                        <dt className="text-ink-2">{x.label}</dt>
                        <dd className="font-medium tabular-nums">{fmt(x)}</dd>
                      </div>
                    ))}
                  </dl>
                ) : (
                  <p className="text-sm text-ink-3">Not trained in this deployment yet.</p>
                )}
              </Card>
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <ShieldCheck size={18} aria-hidden /> Your data
          </h2>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-2">
            <li>Messages and what Aura perceived about them are stored so you can review them and see your insights.</li>
            <li>Webcam frames are analysed in memory and immediately discarded. Only the detected expression is kept with your message.</li>
            <li>Voice recordings are transcribed and discarded. The transcript and voice features are kept, not the audio.</li>
            <li>You can erase Aura’s memory at any time, or delete your account and everything with it.</li>
          </ul>
          <Button variant="secondary" className="text-critical" onClick={() => void removeAccount()}>
            Delete my account and data
          </Button>
        </section>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Aura is not a therapist</h2>
          <p className="text-sm text-ink-2">
            Aura can listen and help you reflect, but it can be wrong and it is not a replacement for professional care. If
            you are thinking about harming yourself, please reach out to someone now:
          </p>
          {resources.length > 0 && <CrisisCard resources={resources} onClose={() => undefined} />}
        </section>

        <section className="space-y-2 border-t border-line pt-6 text-sm text-ink-3">
          <p>
            Built on ESConv (Liu et al., 2021), RAVDESS (Livingstone &amp; Russo, 2018), FER2013, Whisper, wav2vec 2.0, RoBERTa
            and Llama 3.2.
          </p>
          <p>Aura started as a course project at IIT Bhilai with Rahul Raj and was rebuilt end-to-end by Rishi Kharya.</p>
        </section>
      </div>
    </div>
  )
}

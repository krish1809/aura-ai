import { useState, type FormEvent } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router'

import { Button, Logo, Orb } from '../components/ui'
import { ApiError } from '../lib/api'
import { useAuth } from '../lib/auth'

const FEATURES = [
  'Talk, type or show your face. Aura reads how you feel, not just what you say.',
  'It finds what’s behind the feeling and explains every reply.',
  'It remembers what matters to you, and forgets on request.',
]

export function AuthPage({ mode }: { mode: 'login' | 'register' }) {
  const { user, login, register } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const from = (location.state as { from?: string } | null)?.from ?? '/chat'

  if (user) return <Navigate to={from} replace />

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      if (mode === 'login') await login(email, password)
      else await register(name, email, password)
      navigate(from, { replace: true })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not reach Aura. Is the server running?')
    } finally {
      setBusy(false)
    }
  }

  const field = 'w-full rounded-xl border border-line bg-surface px-3.5 py-2.5 text-[15px] text-ink outline-none transition focus:border-accent'

  return (
    <div className="grid min-h-full lg:grid-cols-2">
      <div className="relative hidden overflow-hidden bg-surface-2 lg:flex lg:flex-col lg:justify-between lg:p-12">
        <Logo />
        <Orb className="absolute -right-24 -bottom-24 size-[28rem] opacity-30 blur-3xl" thinking />
        <div className="relative max-w-md space-y-6">
          <h1 className="font-display text-4xl leading-tight font-semibold tracking-tight">
            A companion that listens to how you feel.
          </h1>
          <ul className="space-y-3 text-ink-2">
            {FEATURES.map((f) => (
              <li key={f} className="flex gap-3">
                <Orb className="mt-1.5 size-2.5 shrink-0" />
                {f}
              </li>
            ))}
          </ul>
        </div>
        <p className="relative text-xs text-ink-3">Aura is a supportive companion, not a substitute for professional care.</p>
      </div>

      <div className="flex items-center justify-center p-6">
        <form onSubmit={onSubmit} className="w-full max-w-sm space-y-5">
          <div className="space-y-2">
            <Logo className="lg:hidden" />
            <h2 className="font-display text-2xl font-semibold">{mode === 'login' ? 'Welcome back' : 'Create your space'}</h2>
            <p className="text-sm text-ink-2">
              {mode === 'login' ? 'Sign in to continue your conversations.' : 'Your conversations are private to you.'}
            </p>
          </div>
          {mode === 'register' && (
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">What should Aura call you?</span>
              <input className={field} value={name} onChange={(e) => setName(e.target.value)} required maxLength={80} autoComplete="given-name" />
            </label>
          )}
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">Email</span>
            <input className={field} type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm font-medium">Password</span>
            <input
              className={field}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={mode === 'register' ? 8 : undefined}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
            {mode === 'register' && <span className="text-xs text-ink-3">At least 8 characters.</span>}
          </label>
          {error && (
            <p role="alert" className="rounded-xl bg-critical-soft px-3 py-2 text-sm">
              {error}
            </p>
          )}
          <Button type="submit" className="w-full py-2.5" disabled={busy}>
            {busy ? 'One moment…' : mode === 'login' ? 'Sign in' : 'Create account'}
          </Button>
          <p className="text-center text-sm text-ink-2">
            {mode === 'login' ? 'New to Aura? ' : 'Already have an account? '}
            <Link to={mode === 'login' ? '/register' : '/login'} state={location.state} className="font-medium text-accent hover:underline">
              {mode === 'login' ? 'Create an account' : 'Sign in'}
            </Link>
          </p>
        </form>
      </div>
    </div>
  )
}

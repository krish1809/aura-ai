import clsx from 'clsx'
import {
  BarChart3,
  Brain,
  Info,
  LogOut,
  Menu,
  MessageCircle,
  Monitor,
  Moon,
  Pencil,
  Plus,
  Sun,
  Trash2,
  X,
} from 'lucide-react'
import { useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router'

import { useAuth } from '../lib/auth'
import { ConversationsProvider, useConversations } from '../lib/conversations'
import { MediaProvider } from '../lib/media'
import { useTheme, type Theme } from '../lib/theme'
import { useDialog } from './Dialog'
import { IconButton, Logo } from './ui'

const NAV = [
  { to: '/insights', label: 'Insights', icon: BarChart3 },
  { to: '/memory', label: 'Memory', icon: Brain },
  { to: '/about', label: 'How Aura works', icon: Info },
]

function relativeDay(iso: string): string {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000)
  return days <= 0 ? 'Today' : days === 1 ? 'Yesterday' : days < 7 ? `${days} days ago` : new Date(iso).toLocaleDateString()
}

function ConversationList({ onNavigate }: { onNavigate: () => void }) {
  const { conversations, remove, rename } = useConversations()
  const dialog = useDialog()
  const activeId = useLocation().pathname.match(/^\/chat\/([^/]+)/)?.[1]
  const navigate = useNavigate()

  if (conversations === null) return <p className="px-3 py-2 text-sm text-ink-3">Loading…</p>
  const visible = conversations.filter((c) => c.user_turns > 0 || c.id === activeId)
  if (visible.length === 0) return <p className="px-3 py-2 text-sm text-ink-3">Your conversations will appear here.</p>

  const groups = new Map<string, typeof conversations>()
  for (const c of visible) {
    const day = relativeDay(c.updated_at)
    groups.set(day, [...(groups.get(day) ?? []), c])
  }

  return (
    <div className="space-y-4">
      {[...groups.entries()].map(([day, items]) => (
        <div key={day}>
          <p className="px-3 pb-1 text-xs font-medium text-ink-3">{day}</p>
          <ul>
            {items.map((c) => (
              <li key={c.id} className="group relative">
                <NavLink
                  to={`/chat/${c.id}`}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    clsx(
                      'block truncate rounded-lg py-2 pr-16 pl-3 text-sm transition',
                      isActive ? 'bg-surface-2 font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                    )
                  }
                >
                  {c.title}
                </NavLink>
                <span
                  className={clsx(
                    'absolute top-1/2 right-1 -translate-y-1/2 gap-0.5',
                    // touch screens have no hover: always show there, and on the open conversation
                    activeId === c.id ? 'flex' : 'hidden group-focus-within:flex group-hover:flex [@media(hover:none)]:flex',
                  )}
                >
                  <button
                    className="rounded-md p-1.5 text-ink-3 hover:bg-surface-3 hover:text-ink"
                    aria-label={`Rename ${c.title}`}
                    onClick={async () => {
                      const title = await dialog.prompt({ title: 'Rename conversation', label: 'Title', initial: c.title, confirmLabel: 'Rename' })
                      if (title) await rename(c.id, title)
                    }}
                  >
                    <Pencil size={14} />
                  </button>
                  <button
                    className="rounded-md p-1.5 text-ink-3 hover:bg-surface-3 hover:text-critical"
                    aria-label={`Delete ${c.title}`}
                    onClick={async () => {
                      const ok = await dialog.confirm({
                        title: 'Delete this conversation?',
                        body: 'The messages and what Aura noticed in them will be deleted. This cannot be undone.',
                        confirmLabel: 'Delete',
                        danger: true,
                      })
                      if (!ok) return
                      await remove(c.id)
                      if (activeId === c.id) navigate('/chat')
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}

const THEMES: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'system', label: 'System', icon: Monitor },
  { value: 'dark', label: 'Dark', icon: Moon },
]

function Sidebar({ onNavigate }: { onNavigate: () => void }) {
  const { user, logout } = useAuth()
  const [theme, setTheme] = useTheme()
  const navigate = useNavigate()

  return (
    <div className="flex h-full flex-col gap-4 p-3">
      <div className="flex items-center justify-between px-2 pt-1">
        <Logo />
      </div>
      <button
        onClick={() => {
          navigate('/chat')
          onNavigate()
        }}
        className="flex items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2.5 text-sm font-medium text-ink shadow-xs transition hover:bg-surface-2"
      >
        <Plus size={16} /> New conversation
      </button>
      <nav className="min-h-0 flex-1 overflow-y-auto" aria-label="Conversations">
        <ConversationList onNavigate={onNavigate} />
      </nav>
      <nav className="space-y-0.5 border-t border-line pt-3" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onNavigate}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition',
                isActive ? 'bg-surface-2 font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
              )
            }
          >
            <Icon size={16} /> {label}
          </NavLink>
        ))}
      </nav>
      <div className="flex items-center justify-between gap-2 border-t border-line px-1 pt-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{user?.name}</p>
          <p className="truncate text-xs text-ink-3">{user?.email}</p>
        </div>
        <div className="flex items-center">
          <div className="flex rounded-lg bg-surface-2 p-0.5" role="radiogroup" aria-label="Theme">
            {THEMES.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                role="radio"
                aria-checked={theme === value}
                aria-label={label}
                title={label}
                onClick={() => setTheme(value)}
                className={clsx('rounded-md p-1.5', theme === value ? 'bg-surface text-ink shadow-xs' : 'text-ink-3 hover:text-ink')}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>
          <IconButton label="Sign out" onClick={() => void logout()} className="size-8">
            <LogOut size={16} />
          </IconButton>
        </div>
      </div>
    </div>
  )
}

export function AppShell() {
  const [open, setOpen] = useState(false)
  const location = useLocation()
  const close = () => setOpen(false)

  return (
    <ConversationsProvider>
      <MediaProvider>
      <div className="flex h-full">
        <aside className="hidden w-72 shrink-0 border-r border-line bg-bg md:block">
          <Sidebar onNavigate={close} />
        </aside>
        {open && (
          <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
            <button className="absolute inset-0 bg-black/40" aria-label="Close navigation" onClick={close} />
            <aside className="absolute inset-y-0 left-0 w-[85%] max-w-80 bg-bg shadow-xl">
              <IconButton label="Close navigation" onClick={close} className="absolute top-3 right-3">
                <X size={18} />
              </IconButton>
              <Sidebar onNavigate={close} />
            </aside>
          </div>
        )}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-2 border-b border-line px-3 py-2 md:hidden">
            <IconButton label="Open navigation" onClick={() => setOpen(true)}>
              <Menu size={18} />
            </IconButton>
            <Logo />
            {!location.pathname.startsWith('/chat') && (
              <NavLink to="/chat" className="ml-auto text-ink-2" aria-label="Chat">
                <MessageCircle size={20} />
              </NavLink>
            )}
          </div>
          <Outlet />
        </main>
      </div>
      </MediaProvider>
    </ConversationsProvider>
  )
}

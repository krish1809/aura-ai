import { lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useParams } from 'react-router'

import { AppShell } from './components/AppShell'
import { AuthProvider, RequireAuth } from './lib/auth'
import { AuthPage } from './pages/AuthPage'
import { ChatPage } from './pages/ChatPage'

// Heavier, less-visited pages (charts, force-graph) load on demand.
const InsightsPage = lazy(() => import('./pages/InsightsPage').then((m) => ({ default: m.InsightsPage })))
const MemoryPage = lazy(() => import('./pages/MemoryPage').then((m) => ({ default: m.MemoryPage })))
const AboutPage = lazy(() => import('./pages/AboutPage').then((m) => ({ default: m.AboutPage })))

/** A fresh ChatPage per conversation, so no state leaks between conversations. */
function ChatRoute() {
  const { id } = useParams()
  return <ChatPage key={id ?? 'new'} />
}

const page = (el: ReactNode) => (
  <Suspense fallback={<p className="p-6 text-sm text-ink-3">Loading…</p>}>{el}</Suspense>
)

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<AuthPage mode="login" />} />
          <Route path="/register" element={<AuthPage mode="register" />} />
          <Route
            element={
              <RequireAuth>
                <AppShell />
              </RequireAuth>
            }
          >
            <Route path="/chat" element={<ChatRoute />} />
            <Route path="/chat/:id" element={<ChatRoute />} />
            <Route path="/insights" element={page(<InsightsPage />)} />
            <Route path="/memory" element={page(<MemoryPage />)} />
            <Route path="/about" element={page(<AboutPage />)} />
          </Route>
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router'

import { api } from './api'
import type { User } from './types'

interface AuthState {
  user: User | null | undefined // undefined while the session is being checked
  login: (email: string, password: string) => Promise<void>
  register: (name: string, email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  deleteAccount: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null | undefined>(undefined)

  useEffect(() => {
    api.me().then(setUser, () => setUser(null))
  }, [])

  const login = useCallback(async (email: string, password: string) => setUser(await api.login(email, password)), [])
  const register = useCallback(
    async (name: string, email: string, password: string) => setUser(await api.register(name, email, password)),
    [],
  )
  const logout = useCallback(async () => {
    await api.logout()
    setUser(null)
  }, [])
  const deleteAccount = useCallback(async () => {
    await api.deleteAccount()
    setUser(null)
  }, [])

  const value = useMemo(() => ({ user, login, register, logout, deleteAccount }), [user, login, register, logout, deleteAccount])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user } = useAuth()
  const location = useLocation()
  if (user === undefined) {
    return (
      <div className="grid h-full place-items-center">
        <div className="aura-orb animate-breathe size-10 rounded-full" aria-label="Loading" />
      </div>
    )
  }
  if (user === null) return <Navigate to="/login" replace state={{ from: location.pathname }} />
  return <>{children}</>
}

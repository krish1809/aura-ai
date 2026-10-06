import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark' | 'system'

function read(): Theme {
  try {
    const t = localStorage.getItem('aura-theme')
    return t === 'light' || t === 'dark' ? t : 'system'
  } catch {
    return 'system'
  }
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(read)
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') delete root.dataset.theme
    else root.dataset.theme = theme
    try {
      if (theme === 'system') localStorage.removeItem('aura-theme')
      else localStorage.setItem('aura-theme', theme)
    } catch {
      /* storage unavailable (private mode) - the theme still applies for this visit */
    }
  }, [theme])
  return [theme, setTheme]
}

/** Resolve a CSS custom property to a concrete colour (canvas APIs cannot read var()). */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

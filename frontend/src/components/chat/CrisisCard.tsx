import { HeartHandshake, Phone, X } from 'lucide-react'

import type { CrisisResource } from '../../lib/types'
import { IconButton } from '../ui'

function href(contact: string): string | undefined {
  if (contact.startsWith('http')) return contact
  const digits = contact.match(/[\d-]{3,}/)?.[0]?.replace(/-/g, '')
  return digits ? `tel:${digits}` : undefined
}

export function CrisisCard({ resources, onClose }: { resources: CrisisResource[]; onClose: () => void }) {
  return (
    <div role="alert" className="relative rounded-2xl border border-critical/30 bg-critical-soft p-4">
      <IconButton label="Dismiss" onClick={onClose} className="absolute top-2 right-2 size-8">
        <X size={16} />
      </IconButton>
      <div className="flex items-start gap-3 pr-8">
        <HeartHandshake className="mt-0.5 shrink-0 text-critical" size={22} aria-hidden />
        <div className="space-y-2">
          <p className="font-semibold">You don’t have to go through this alone.</p>
          <p className="text-sm text-ink-2">These services are free and confidential. Please reach out to one now, or to someone you trust.</p>
          <ul className="grid gap-1.5 sm:grid-cols-2">
            {resources.map((r) => {
              const link = href(r.contact)
              return (
                <li key={r.name} className="rounded-xl bg-surface px-3 py-2 text-sm">
                  <p className="text-xs text-ink-3">{r.region}</p>
                  <p className="font-medium">{r.name}</p>
                  {link ? (
                    <a href={link} className="inline-flex items-center gap-1 text-critical underline-offset-2 hover:underline">
                      <Phone size={12} aria-hidden /> {r.contact}
                    </a>
                  ) : (
                    <p>{r.contact}</p>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      </div>
    </div>
  )
}

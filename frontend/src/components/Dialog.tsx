import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'

import { Button } from './ui'

interface ConfirmOptions {
  title: string
  body?: string
  confirmLabel?: string
  danger?: boolean
}
interface PromptOptions {
  title: string
  label: string
  initial?: string
  confirmLabel?: string
}
type Request =
  | ({ kind: 'confirm' } & ConfirmOptions & { resolve: (ok: boolean) => void })
  | ({ kind: 'prompt' } & PromptOptions & { resolve: (value: string | null) => void })

interface DialogApi {
  confirm: (o: ConfirmOptions) => Promise<boolean>
  prompt: (o: PromptOptions) => Promise<string | null>
}

const DialogContext = createContext<DialogApi | null>(null)

/** In-app replacements for window.confirm / window.prompt, built on the native <dialog>. */
export function DialogProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<Request | null>(null)
  const [value, setValue] = useState('')
  const ref = useRef<HTMLDialogElement>(null)

  const confirm = useCallback(
    (o: ConfirmOptions) => new Promise<boolean>((resolve) => setRequest({ kind: 'confirm', ...o, resolve })),
    [],
  )
  const prompt = useCallback(
    (o: PromptOptions) =>
      new Promise<string | null>((resolve) => {
        setValue(o.initial ?? '')
        setRequest({ kind: 'prompt', ...o, resolve })
      }),
    [],
  )

  useEffect(() => {
    if (request && ref.current && !ref.current.open) ref.current.showModal()
  }, [request])

  const finish = (ok: boolean) => {
    if (!request) return
    if (request.kind === 'confirm') request.resolve(ok)
    else request.resolve(ok && value.trim() ? value.trim() : null)
    ref.current?.close()
    setRequest(null)
  }

  return (
    <DialogContext.Provider value={{ confirm, prompt }}>
      {children}
      <dialog
        ref={ref}
        onCancel={(e) => {
          e.preventDefault()
          finish(false)
        }}
        className="m-auto w-[min(26rem,calc(100%-2rem))] rounded-2xl border border-line bg-surface p-0 text-ink shadow-2xl backdrop:bg-black/40"
        aria-labelledby="aura-dialog-title"
      >
        {request && (
          <form
            method="dialog"
            className="space-y-4 p-5"
            onSubmit={(e) => {
              e.preventDefault()
              finish(true)
            }}
          >
            <h2 id="aura-dialog-title" className="text-lg font-semibold">
              {request.title}
            </h2>
            {request.kind === 'confirm' && request.body && <p className="text-sm text-ink-2">{request.body}</p>}
            {request.kind === 'prompt' && (
              <label className="block space-y-1.5">
                <span className="text-sm text-ink-2">{request.label}</span>
                <input
                  autoFocus
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  maxLength={120}
                  className="w-full rounded-xl border border-line bg-surface px-3 py-2 text-ink outline-none focus:border-accent"
                />
              </label>
            )}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="ghost" onClick={() => finish(false)}>
                Cancel
              </Button>
              <Button
                type="submit"
                autoFocus={request.kind === 'confirm'}
                variant={request.kind === 'confirm' && request.danger ? 'danger' : 'primary'}
              >
                {request.confirmLabel ?? 'OK'}
              </Button>
            </div>
          </form>
        )}
      </dialog>
    </DialogContext.Provider>
  )
}

export function useDialog(): DialogApi {
  const ctx = useContext(DialogContext)
  if (!ctx) throw new Error('useDialog must be used inside <DialogProvider>')
  return ctx
}

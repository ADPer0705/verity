/**
 * Data-fetching hooks.
 *
 * Deliberately dependency-free (no react-query): the surface is small, and a
 * forensics console benefits from explicit, inspectable refresh semantics
 * more than it does from a cache layer.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api } from './client'
import type { Job, JobLogLine } from './types'

export interface AsyncState<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/** Run `fetcher` on mount and whenever `deps` change. */
export function useAsync<T>(fetcher: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    fetcher()
      .then((value) => {
        if (!alive.current) return
        setData(value)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (!alive.current) return
        setError(cause instanceof ApiError ? cause.message : String(cause))
      })
      .finally(() => {
        if (alive.current) setLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, error, loading, reload }
}

/** Poll `fetcher` on an interval; pauses while the tab is hidden. */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled = true,
  deps: unknown[] = [],
): AsyncState<T> {
  const state = useAsync(fetcher, deps)
  const { reload } = state

  useEffect(() => {
    if (!enabled) return
    const id = window.setInterval(() => {
      if (document.visibilityState === 'visible') reload()
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [enabled, intervalMs, reload])

  return state
}

export interface JobStream {
  job: Job | null
  logs: JobLogLine[]
  error: string | null
  finished: boolean
}

/**
 * Follow one job to completion.
 *
 * Prefers server-sent events; falls back to polling when EventSource is
 * unavailable or the stream drops (some corporate proxies buffer SSE into
 * uselessness, and a stalled progress bar during a 40-minute detector run is
 * worse than a slightly chattier poll).
 */
export function useJobStream(jobId: string | null): JobStream {
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<JobLogLine[]>([])
  const [error, setError] = useState<string | null>(null)
  const seenSeq = useRef(0)

  useEffect(() => {
    setJob(null)
    setLogs([])
    setError(null)
    seenSeq.current = 0
    if (!jobId) return

    let cancelled = false
    let source: EventSource | null = null
    let pollTimer: number | null = null

    const absorb = (payload: Job) => {
      if (cancelled) return
      setJob(payload)
      if (payload.logs?.length) {
        setLogs((previous) => {
          const fresh = payload.logs!.filter((line) => line.seq > seenSeq.current)
          if (!fresh.length) return previous
          seenSeq.current = Math.max(seenSeq.current, ...fresh.map((l) => l.seq))
          return [...previous, ...fresh]
        })
      } else {
        seenSeq.current = Math.max(seenSeq.current, payload.log_seq ?? 0)
      }
    }

    const poll = () => {
      api
        .job(jobId, seenSeq.current)
        .then((payload) => {
          absorb(payload)
          if (cancelled) return
          if (payload.status === 'pending' || payload.status === 'running') {
            pollTimer = window.setTimeout(poll, 900)
          }
        })
        .catch((cause: unknown) => {
          if (!cancelled) setError(cause instanceof ApiError ? cause.message : String(cause))
        })
    }

    if (typeof EventSource !== 'undefined') {
      source = new EventSource(`/api/jobs/${jobId}/events`)
      source.onmessage = (event) => {
        try {
          absorb(JSON.parse(event.data) as Job)
        } catch {
          /* ignore malformed frame; the next one supersedes it */
        }
      }
      source.addEventListener('done', () => source?.close())
      source.onerror = () => {
        source?.close()
        source = null
        if (!cancelled) poll()
      }
    } else {
      poll()
    }

    // One immediate read so the panel is populated before the first event.
    api.job(jobId, 0).then(absorb).catch(() => undefined)

    return () => {
      cancelled = true
      source?.close()
      if (pollTimer) window.clearTimeout(pollTimer)
    }
  }, [jobId])

  const finished =
    job !== null && ['succeeded', 'failed', 'cancelled'].includes(job.status)

  return { job, logs, error, finished }
}

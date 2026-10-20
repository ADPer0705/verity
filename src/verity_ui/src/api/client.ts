/**
 * Thin fetch wrapper around the Verity API.
 *
 * Errors surface the server's `detail` string verbatim: the backend writes
 * those to be read by an investigator (missing weights, absent ffprobe,
 * unknown detector), so swallowing them behind a generic message would throw
 * away the most useful part of the response.
 */

import type {
  CaseDetail,
  CaseSummary,
  Coverage,
  DetectorAssets,
  DetectorCatalog,
  Finding,
  Consensus,
  IntegrityCheck,
  Job,
  SystemStatus,
  UploadResult,
} from './types'

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
    })
  } catch (cause) {
    throw new ApiError(0, `Cannot reach the Verity API. Is the server running? (${String(cause)})`)
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
      else if (Array.isArray(body?.detail)) detail = body.detail.map((d: any) => d.msg).join('; ')
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new ApiError(response.status, detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  body: JSON.stringify(body),
})

export const api = {
  // -- system ------------------------------------------------------- //
  system: () => request<SystemStatus>('/api/system'),
  updateSettings: (patch: Record<string, unknown>) =>
    request<{ settings: SystemStatus['settings']; readiness: SystemStatus['readiness'] }>(
      '/api/system/settings',
      { method: 'PUT', body: JSON.stringify(patch) },
    ),

  // -- detectors ---------------------------------------------------- //
  detectors: () => request<DetectorCatalog>('/api/detectors'),
  coverage: () => request<Coverage>('/api/detectors/coverage'),
  setupDetectors: (detectors: string[] | null, force = false) =>
    request<Job>('/api/detectors/setup', json({ detectors, force })),

  // -- cases -------------------------------------------------------- //
  cases: () => request<{ count: number; cases: CaseSummary[] }>('/api/media'),
  case: (mediaId: string) => request<CaseDetail>(`/api/media/${mediaId}`),
  deleteCase: (mediaId: string) =>
    request<{ media_id: string; removed: string[] }>(`/api/media/${mediaId}?confirm=true`, {
      method: 'DELETE',
    }),

  upload: (file: File, onProgress?: (fraction: number) => void) =>
    new Promise<UploadResult>((resolve, reject) => {
      // XHR rather than fetch: upload progress matters for multi-hundred-MB
      // video, and fetch still has no portable way to report it.
      const form = new FormData()
      form.append('file', file)

      const xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/media')
      xhr.responseType = 'json'

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.response as UploadResult)
        } else {
          const detail = (xhr.response as any)?.detail ?? `${xhr.status} ${xhr.statusText}`
          reject(new ApiError(xhr.status, String(detail)))
        }
      }
      xhr.onerror = () => reject(new ApiError(0, 'Upload failed — the server is unreachable.'))
      xhr.onabort = () => reject(new ApiError(0, 'Upload cancelled.'))
      xhr.send(form)
    }),

  verify: (mediaId: string) =>
    request<IntegrityCheck>(`/api/media/${mediaId}/verify`, { method: 'POST' }),

  findings: (mediaId: string) =>
    request<{ media_id: string; consensus: Consensus; findings: Finding[] }>(
      `/api/media/${mediaId}/findings`,
    ),

  detectorAssets: (mediaId: string, detectorId: string) =>
    request<DetectorAssets>(`/api/media/${mediaId}/detectors/${detectorId}/assets`),

  detectorLogs: (mediaId: string, detectorId: string) =>
    request<{ stdout: string; stderr: string }>(
      `/api/media/${mediaId}/detectors/${detectorId}/logs`,
    ),

  detectorRaw: (mediaId: string, detectorId: string) =>
    request<Record<string, unknown>>(`/api/media/${mediaId}/detectors/${detectorId}/raw`),

  // -- operations (each returns a job) ------------------------------ //
  run: (body: {
    media_id: string
    detectors?: string[] | null
    extractors?: string[] | null
    format?: 'pdf' | 'html'
    skip_report?: boolean
    include_untrained?: boolean
  }) => request<Job>('/api/run', json(body)),

  analyze: (mediaId: string, detectors?: string[] | null, includeUntrained = false) =>
    request<Job>(
      '/api/analyze',
      json({ media_id: mediaId, detectors, include_untrained: includeUntrained }),
    ),

  extract: (mediaId: string, extractors?: string[] | null) =>
    request<Job>('/api/extract', json({ media_id: mediaId, extractors })),

  compileFindings: (mediaId: string) =>
    request<Job>('/api/compile', json({ media_id: mediaId })),

  report: (mediaId: string, format: 'pdf' | 'html') =>
    request<Job>('/api/report', json({ media_id: mediaId, format })),

  // -- jobs --------------------------------------------------------- //
  jobs: (params?: { limit?: number; mediaId?: string }) => {
    const query = new URLSearchParams()
    if (params?.limit) query.set('limit', String(params.limit))
    if (params?.mediaId) query.set('media_id', params.mediaId)
    const suffix = query.toString() ? `?${query}` : ''
    return request<{ active: number; count: number; jobs: Job[] }>(`/api/jobs${suffix}`)
  },
  job: (jobId: string, since = 0) => request<Job>(`/api/jobs/${jobId}?since=${since}`),
  cancelJob: (jobId: string) => request<unknown>(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),
}

/** URL for an artifact on disk, served through the confined file endpoint. */
export const fileUrl = (mediaId: string, path: string) =>
  `/api/media/${mediaId}/files?path=${encodeURIComponent(path)}`

export const mediaFileUrl = (mediaId: string, download = false) =>
  `/api/media/${mediaId}/file${download ? '?download=true' : ''}`

export const reportUrl = (mediaId: string, format: 'pdf' | 'html', download = false) =>
  `/api/media/${mediaId}/report?format=${format}${download ? '&download=true' : ''}`

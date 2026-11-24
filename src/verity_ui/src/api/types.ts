/**
 * Shapes returned by the Verity API.
 *
 * These mirror the pydantic models in `verity.models` / `verity.analyze.models`
 * as serialised by `verity_api.service`. Fields that the pipeline leaves
 * optional are optional here too — the UI is expected to render partial cases
 * (ingested but not analysed, analysed but not detected, and so on).
 */

export type MediaType = 'image' | 'video' | 'audio' | 'unknown'
export type Classification = 'REAL' | 'FAKE' | 'UNCERTAIN'

export type ConsensusBand =
  | 'no_analysis'
  | 'strong_manipulation'
  | 'likely_manipulation'
  | 'inconclusive'
  | 'likely_authentic'
  | 'strong_authentic'

export interface Consensus {
  band: ConsensusBand
  total: number
  fake: number
  real: number
  uncertain: number
  agreement: number
  mean_fake_probability: number | null
  disagreement: boolean
  leaning_split: boolean
  leans_fake?: number
  leans_real?: number
  /** Findings from untrained-head detectors that were left out of the vote. */
  untrained_excluded?: number
}

/** trained | backbone-only | random-init | untrained — anything but "trained" is not evidence. */
export type DetectorTier = 'trained' | 'backbone-only' | 'random-init' | 'untrained'

export interface ReportFile {
  path: string
  size_bytes: number
  generated_at: string
}

export interface ReportFiles {
  html: ReportFile | null
  pdf: ReportFile | null
}

export interface CaseSummary {
  media_id: string
  file_name: string
  media_type: MediaType
  mime_type: string
  size_bytes: number
  checksum: string
  ingested_at: string | null
  has_analysis: boolean
  has_selection: boolean
  has_forensics?: boolean
  forensics_counts?: Record<string, number> | null
  detectors_selected: number
  detectors_excluded: number
  findings_count: number
  aggregated_score: number | null
  consensus: Consensus
  reports: ReportFiles
  face_count: number | null
  duration_seconds: number | null
  width: number | null
  height: number | null
}

export interface MediaCharacteristics {
  media_id: string
  media_type: MediaType
  format: string
  duration_seconds: number | null
  fps: number | null
  width: number | null
  height: number | null
  frame_count: number | null
  video_codec: string
  has_audio_track: boolean
  audio_codec: string
  face_present: boolean
  face_count: number
  sampled_frames: number
  max_face_size: number
}

export interface FramefaceCount {
  frame: number
  face_count: number
  max_face_size: number
}

export interface MediaAnalysis {
  media: MediaCharacteristics
  faces_per_frame: FramefaceCount[]
}

export interface SelectionDecision {
  detector_id: string
  detector_name: string
  included: boolean
  reasons: string[]
}

export interface Selection {
  media_type: MediaType
  decisions: SelectionDecision[]
  selected: string[]
}

export type DetectorUnit = 'face' | 'frame' | 'clip' | 'window'
export type DetectorInputMode =
  | 'face_crop'
  | 'face_clip'
  | 'full_frame'
  | 'full_clip'
  | 'audio'
  | 'av_clip'

export interface PerFace {
  face_id?: string
  frame_idx?: number
  face_index?: number
  bbox?: string | number[]
  fake_prob?: number
  start_s?: number
  end_s?: number
  region_energy?: Record<string, number | RegionEnergy>
  aux?: Record<string, number>
  info?: Record<string, unknown>
  [key: string]: unknown
}

export interface RegionEnergy {
  residual_pct?: number
  attention_pct?: number
  face_count?: number
}

export interface Finding {
  detector_id: string
  detector_name: string
  detector_version: string
  tier: DetectorTier
  trained: boolean
  classification: Classification
  confidence: number
  mean_fake_probability: number
  faces_analyzed: number | null
  unit?: DetectorUnit
  input_mode?: DetectorInputMode
  reasoning: {
    summary: string
    confidence_factors: string[]
    evidence: string[]
  }
  stats: {
    min?: number
    max?: number
    mean?: number
    median?: number
    std?: number
    count?: number
    threshold?: number
    above_threshold?: number
    below_threshold?: number
    high_confidence_fake?: number
    high_confidence_real?: number
    [key: string]: unknown
  }
  sampling: { sample_interval?: number; coverage_pct?: number; [key: string]: unknown }
  region_aggregate: Record<string, RegionEnergy>
  aux_aggregate?: Record<string, number>
  audio?: {
    sample_rate?: number
    duration_seconds?: number
    window_seconds?: number
    hop_seconds?: number
    window_frames?: number
    hop_frames?: number
    fps?: number
    windows?: number
    error?: string
  } | null
  model?: {
    name?: string
    version?: string
    architecture?: string
    parameters?: number | null
    pretrained?: string
  }
  weights?: {
    status?: string
    file?: string | null
    note?: string
  }
  device?: string
  timing_seconds?: {
    model_load?: number
    face_detection?: number
    preprocess?: number
    inference?: number
    visualization?: number
    total?: number
  }
  total_video_frames?: number
  reconstruction?: {
    available?: boolean
    evidence?: string
  } | null
  per_face: PerFace[]
  timestamp: string
  output_dir: string | null
  has_logs: boolean
  /** Manipulation families the detector was trained to recognise. */
  targets?: string[] | null
}

export type ForensicSeverity = 'info' | 'notice' | 'warning'

export interface ForensicObservation {
  analyzer: string
  severity: ForensicSeverity
  statement: string
  evidence: Record<string, unknown>
}

export interface ForensicAnalyzerResult {
  analyzer: string
  label: string
  method: string
  ok: boolean
  observations: ForensicObservation[]
  metrics: Record<string, unknown>
  artifacts: string[]
}

export interface ForensicsReport {
  media_id: string
  media_type: MediaType
  results: ForensicAnalyzerResult[]
  counts?: Record<ForensicSeverity, number>
}

export interface FileEntry {
  name: string
  path: string
  size_bytes: number
  modified_at: string
  suffix: string
}

export interface ArtifactGroup {
  group: string
  count: number
  files: FileEntry[]
}

export interface CaseDetail {
  media_id: string
  summary: CaseSummary
  media: {
    original_path: string
    file_name: string
    media_type: MediaType
    mime_type: string
    checksum: string
    size_bytes: number
    metadata: Record<string, unknown>
    working_copy: string
  }
  analysis: MediaAnalysis | null
  selection: Selection | null
  findings: Finding[]
  aggregated_score: number | null
  consensus: Consensus
  artifacts: ArtifactGroup[]
  forensics: ForensicsReport | null
  reports: ReportFiles
}

export interface DetectorWeights {
  file: string
  kind: string
  license?: string
  url: string
  size: number | null
  note: string
  self_contained: boolean
  available: boolean
}

export interface DetectorEntry {
  detector_id: string
  name: string
  version: string
  description: string
  media_types: string[]
  faces_required: boolean
  video_mode: boolean
  clip_size: number | null
  min_face: number
  resolution: number
  explainability: boolean
  tier: DetectorTier
  trained: boolean
  /** Manipulation families (taxonomy ids) this detector targets. */
  targets: string[]
  weights: DetectorWeights
}

/** One manipulation family and the detectors, by tier, that target it. */
export interface CoverageFamily {
  id: string
  label: string
  description: string
  modalities: string[]
  trained: string[]
  untrained: string[]
  covered: boolean
}

export interface Coverage {
  families: CoverageFamily[]
  covered: number
  total: number
}

export interface DetectorCatalog {
  detectors_dir: string
  detectors_dir_present: boolean
  count: number
  weights_ready: number
  detectors: DetectorEntry[]
}

export interface DetectorAssets {
  exists: boolean
  output_dir?: string
  timeline: FileEntry | null
  full_frames: FileEntry[]
  faces: FileEntry[]
  frames: FileEntry[]
  attention_maps: FileEntry[]
  reconstructions: FileEntry[]
}

export interface IntegrityCheck {
  media_id: string
  expected_checksum: string
  actual_checksum: string
  matches: boolean
  working_copy: string
  working_copy_size: number
  recorded_size: number
  size_matches: boolean
  original_path: string
  original_state: 'present' | 'absent'
  original_matches: boolean | null
  verified_at: string
}

export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type StepStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped'

export interface JobStep {
  key: string
  label: string
  status: StepStatus
  detail: string
  started_at: string | null
  finished_at: string | null
}

export interface JobLogLine {
  seq: number
  ts: string
  level: 'info' | 'warn' | 'error'
  message: string
}

export interface Job {
  id: string
  kind: string
  label: string
  media_ids: string[]
  params: Record<string, unknown>
  status: JobStatus
  steps: JobStep[]
  logs?: JobLogLine[]
  result: unknown
  error: string | null
  traceback: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  log_seq: number
}

export interface Blocker {
  capability: string
  message: string
}

export interface SystemStatus {
  settings: {
    detectors_dir: string
    workspace: string
    output_dir: string
    extractors: string[]
    offline: boolean
    config_path: string | null
  }
  available_extractors: string[]
  environment: {
    detectors_dir_present: boolean
    detector_count: number
    detector_runner: string | null
    ffprobe: string | null
    ffmpeg: string | null
    uv: string | null
    weasyprint: boolean
    /** Why PDF rendering is unavailable, when `weasyprint` is false. */
    weasyprint_error?: string
    opencv: boolean
    hw_profile: string
  }
  readiness: {
    can_ingest: boolean
    can_analyze: boolean
    can_detect: boolean
    can_setup: boolean
    can_report_html: boolean
    can_report_pdf: boolean
    blockers: Blocker[]
  }
  detectors_registered: number
  active_jobs: number
}

export interface UploadResult {
  media_id: string
  media_type: MediaType
  mime_type: string
  checksum: string
  size_bytes: number
  file_name: string
  copy_path: string
  case: CaseSummary
}

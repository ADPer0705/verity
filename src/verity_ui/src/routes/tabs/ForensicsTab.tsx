/**
 * Model-free forensic examinations (metadata, compression/ELA, noise/FFT,
 * video temporal consistency, and audio spectral analysis).
 *
 * Unlike the neural detectors, every observation here is a direct deterministic
 * measurement on the file (EXIF/container tags, JPEG 8x8 grid & Error Level
 * Analysis, median-filter noise residual & 2D FFT peaks, frame-to-frame SSIM/luminance
 * jumps, and log-mel audio spectrogram statistics).
 */

import { useMemo, useState } from 'react'
import {
  Badge,
  Blockquote,
  Button,
  Group,
  Paper,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Table,
  Text,
} from '@mantine/core'
import {
  IconAlertTriangle,
  IconCheck,
  IconInfoCircle,
  IconMicroscope,
  IconPlayerPlay,
} from '@tabler/icons-react'

import type {
  CaseDetail,
  ForensicAnalyzerResult,
  ForensicObservation,
  ForensicSeverity,
} from '@/api/types'
import { ImageGallery } from '@/components/ImageGallery'
import type { GalleryImage } from '@/components/ImageGallery'
import { EmptyState, MetricGrid, SectionTitle } from '@/components/Primitives'

const SEVERITY_ORDER: Record<ForensicSeverity, number> = {
  warning: 0,
  notice: 1,
  info: 2,
}

const SEVERITY_COLOR: Record<ForensicSeverity, string> = {
  warning: 'red',
  notice: 'orange',
  info: 'blue',
}

const IMAGE_CAPTIONS: Record<string, string> = {
  'ela.png': 'Error Level Analysis (brighter regions change more on JPEG re-save)',
  'ela_overlay.png': 'ELA heatmap overlaid on the image',
  'blockiness.png': 'Local 8×8 JPEG blockiness map (red = textured cells missing the grid)',
  'noise_residual.png': 'High-frequency noise residual (image minus median-filtered copy)',
  'noise_variance.png':
    'Local noise variance map (cyan = unusually smooth, yellow = unusually noisy)',
  'spectrum.png': 'Log-magnitude 2D Fourier spectrum (circles mark isolated periodic peaks)',
  'timeline.png': 'Frame-difference, luminance and face-size temporal profile',
  'spectrogram.png': 'Log-mel spectrogram of the audio stream',
}

function formatEvidenceValue(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/\.?0+$/, '')
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (value === null || value === undefined) return '—'
  if (Array.isArray(value)) return value.slice(0, 4).map(formatEvidenceValue).join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function EvidenceChips({ evidence }: { evidence: Record<string, unknown> }) {
  const entries = Object.entries(evidence).filter(
    ([, val]) => val !== null && val !== undefined && typeof val !== 'object',
  )
  if (!entries.length) {
    return (
      <Text component="span" fz="xs" c="dimmed">
        —
      </Text>
    )
  }
  return (
    <Group gap={4} wrap="wrap">
      {entries.slice(0, 8).map(([key, val]) => (
        <Badge
          key={key}
          variant="light"
          color="gray"
          size="xs"
          ff="monospace"
          style={{ textTransform: 'none' }}
        >
          {key}={formatEvidenceValue(val)}
        </Badge>
      ))}
    </Group>
  )
}

function AnalyzerSection({
  mediaId,
  result,
  severityFilter,
}: {
  mediaId: string
  result: ForensicAnalyzerResult
  severityFilter: 'all' | ForensicSeverity
}) {
  const observations = useMemo(() => {
    const filtered = result.observations.filter(
      (obs) => severityFilter === 'all' || obs.severity === severityFilter,
    )
    return [...filtered].sort(
      (a, b) =>
        SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] ||
        a.statement.localeCompare(b.statement),
    )
  }, [result.observations, severityFilter])

  const images: GalleryImage[] = useMemo(() => {
    return (result.artifacts ?? [])
      .filter((path) => /\.(png|jpg|jpeg|webp)$/i.test(path))
      .map((path) => {
        const name = path.split('/').pop() ?? path
        return {
          name,
          path,
          size_bytes: 0,
          modified_at: '',
          suffix: name.split('.').pop()?.toLowerCase() ?? 'png',
          caption: IMAGE_CAPTIONS[name] ?? name.replace(/\.[^.]+$/, ''),
          badge: name.replace(/\.[^.]+$/, ''),
        }
      })
  }, [result.artifacts])

  const warnings = result.observations.filter((o) => o.severity === 'warning').length
  const notices = result.observations.filter((o) => o.severity === 'notice').length
  const infos = result.observations.filter((o) => o.severity === 'info').length

  if (severityFilter !== 'all' && observations.length === 0) {
    return null
  }

  return (
    <Paper withBorder p="md" className="vy-card">
      <Group justify="space-between" align="flex-start" mb="xs" wrap="wrap">
        <div>
          <Group gap="xs" align="center">
            <Text fw={600} fz="md">
              {result.label}
            </Text>
            <Badge variant="outline" color="gray" size="xs" ff="monospace">
              {result.analyzer}
            </Badge>
            {!result.ok ? (
              <Badge color="red" variant="filled" size="xs">
                analyzer failed
              </Badge>
            ) : null}
          </Group>
        </div>
        <Group gap={6}>
          {warnings > 0 ? (
            <Badge color="red" size="sm">
              {warnings} warning{warnings === 1 ? '' : 's'}
            </Badge>
          ) : null}
          {notices > 0 ? (
            <Badge color="orange" size="sm">
              {notices} notice{notices === 1 ? '' : 's'}
            </Badge>
          ) : null}
          {infos > 0 ? (
            <Badge color="gray" variant="light" size="sm">
              {infos} info
            </Badge>
          ) : null}
        </Group>
      </Group>

      <Blockquote p="xs" mb="md" color="gray">
        <Text fz="xs" c="dimmed" lh={1.5}>
          {result.method}
        </Text>
      </Blockquote>

      {observations.length > 0 ? (
        <Table.ScrollContainer minWidth={560} mb={images.length ? 'md' : 0}>
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th w={100}>Severity</Table.Th>
                <Table.Th>Observation</Table.Th>
                <Table.Th w="38%">Supporting measurements</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {observations.map((obs: ForensicObservation, idx: number) => (
                <Table.Tr key={`${obs.analyzer}-${idx}`}>
                  <Table.Td>
                    <Badge
                      color={SEVERITY_COLOR[obs.severity]}
                      variant={obs.severity === 'warning' ? 'filled' : 'light'}
                      size="sm"
                    >
                      {obs.severity}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Text fz="sm" lh={1.45}>
                      {obs.statement}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <EvidenceChips evidence={obs.evidence} />
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      ) : (
        <Text fz="sm" c="dimmed" mb={images.length ? 'md' : 0}>
          No observations recorded for this filter.
        </Text>
      )}

      {images.length > 0 ? (
        <div>
          <SectionTitle>Diagnostic figures ({images.length})</SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Click any figure to inspect at full resolution.
          </Text>
          <ImageGallery
            mediaId={mediaId}
            images={images}
            layout={
              images.some((img) => img.name === 'timeline.png' || img.name === 'spectrogram.png')
                ? 'wide'
                : 'maps'
            }
          />
        </div>
      ) : null}
    </Paper>
  )
}

export function ForensicsTab({
  detail,
  busy = false,
  onRunForensics,
}: {
  detail: CaseDetail
  busy?: boolean
  onRunForensics?: () => void
}) {
  const forensics = detail.forensics
  const [severityFilter, setSeverityFilter] = useState<'all' | ForensicSeverity>('all')

  if (!forensics || !forensics.results?.length) {
    return (
      <EmptyState
        icon={<IconMicroscope size={26} />}
        title="Model-free forensic analyses have not run yet"
        body="Forensic analyzers inspect container metadata, JPEG compression / Error Level Analysis (ELA), high-frequency noise residuals, 2D Fourier spectra, frame-to-frame temporal consistency, and audio spectrograms without relying on neural network weights."
        action={
          onRunForensics ? (
            <Button
              leftSection={<IconPlayerPlay size={16} />}
              disabled={busy}
              onClick={onRunForensics}
            >
              Run forensic analyses
            </Button>
          ) : undefined
        }
      />
    )
  }

  const allObservations = forensics.results.flatMap((r) => r.observations)
  const warningCount =
    forensics.counts?.warning ?? allObservations.filter((o) => o.severity === 'warning').length
  const noticeCount =
    forensics.counts?.notice ?? allObservations.filter((o) => o.severity === 'notice').length
  const infoCount =
    forensics.counts?.info ?? allObservations.filter((o) => o.severity === 'info').length
  const figureCount = forensics.results.reduce((acc, r) => acc + (r.artifacts?.length ?? 0), 0)

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start" wrap="wrap">
        <div>
          <SectionTitle>Model-free signal & metadata forensics</SectionTitle>
          <Text fz="sm" c="dimmed" maw="58rem">
            Deterministic physical and statistical checks that complement the neural detectors. A{' '}
            <b>warning</b> indicates an internal contradiction or strong structural anomaly, a{' '}
            <b>notice</b> departs from an untouched camera capture, and <b>info</b> records a
            verified baseline measurement.
          </Text>
        </div>
        {onRunForensics ? (
          <Button
            variant="default"
            size="xs"
            leftSection={<IconPlayerPlay size={14} />}
            disabled={busy}
            onClick={onRunForensics}
          >
            Re-run forensics
          </Button>
        ) : null}
      </Group>

      <MetricGrid
        metrics={[
          {
            label: 'Warnings',
            value: (
              <Group gap={6} align="center">
                <Text
                  component="span"
                  fz="xl"
                  fw={600}
                  c={warningCount > 0 ? 'red' : undefined}
                >
                  {warningCount}
                </Text>
                {warningCount > 0 ? <IconAlertTriangle size={18} color="var(--mantine-color-red-6)" /> : null}
              </Group>
            ),
            hint: 'Internal contradictions or strong anomalies',
          },
          {
            label: 'Notices',
            value: (
              <Text
                component="span"
                fz="xl"
                fw={600}
                c={noticeCount > 0 ? 'orange' : undefined}
              >
                {noticeCount}
              </Text>
            ),
            hint: 'Departures from untouched capture',
          },
          {
            label: 'Baseline facts',
            value: (
              <Group gap={6} align="center">
                <Text component="span" fz="xl" fw={600}>
                  {infoCount}
                </Text>
                <IconInfoCircle size={16} color="var(--mantine-color-dimmed)" />
              </Group>
            ),
            hint: `Across ${forensics.results.length} active analyzer(s)`,
          },
          {
            label: 'Diagnostic figures',
            value: (
              <Group gap={6} align="center">
                <Text component="span" fz="xl" fw={600}>
                  {figureCount}
                </Text>
                {figureCount > 0 ? <IconCheck size={16} color="var(--mantine-color-teal-6)" /> : null}
              </Group>
            ),
            hint: 'ELA, noise residual, FFT spectrum, timeline & spectrogram',
          },
        ]}
      />

      <Group justify="space-between" align="center" wrap="wrap">
        <Text fz="xs" c="dimmed">
          Showing analyzers for <b>{detail.media.media_type}</b> media
        </Text>
        <SegmentedControl
          size="xs"
          value={severityFilter}
          onChange={(val) => setSeverityFilter(val as 'all' | ForensicSeverity)}
          data={[
            { value: 'all', label: `All (${allObservations.length})` },
            { value: 'warning', label: `Warnings (${warningCount})` },
            { value: 'notice', label: `Notices (${noticeCount})` },
            { value: 'info', label: `Info (${infoCount})` },
          ]}
        />
      </Group>

      <SimpleGrid cols={1} spacing="md">
        {forensics.results.map((result) => (
          <AnalyzerSection
            key={result.analyzer}
            mediaId={detail.media_id}
            result={result}
            severityFilter={severityFilter}
          />
        ))}
      </SimpleGrid>
    </Stack>
  )
}

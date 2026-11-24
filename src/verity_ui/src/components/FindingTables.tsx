/** Tabular detail shared by the Findings and Evidence tabs. */

import { useMemo, useState } from 'react'
import { Badge, Button, Group, Paper, Progress, SimpleGrid, Table, Text } from '@mantine/core'

import type { DetectorInputMode, DetectorUnit, Finding, PerFace } from '@/api/types'
import { ProbabilityBar } from '@/components/Primitives'

const PER_FACE_PAGE = 40

const AUX_LABELS: Record<string, string> = {
  deepfake: 'Deepfake branch P(fake)',
  blend: 'Blending-boundary branch',
  identity_inconsistency: 'Identity inconsistency branch',
  clip_branch: 'CLIP query head P(fake)',
  vla_branch: 'Vision-Language identity head P(fake)',
  sd1_lpips: 'Stable Diffusion 1.1 VAE LPIPS distance',
  sd2_lpips: 'Stable Diffusion 2.1 VAE LPIPS distance',
  kd2_lpips: 'Kandinsky 2.1 MoVQ LPIPS distance',
  min_lpips: 'Minimum LDM reconstruction LPIPS error',
  av_offset_frames: 'Audio-visual lag offset (frames)',
  av_offset_seconds: 'Audio-visual lag offset (seconds)',
  min_distance: 'SyncNet minimum AV embedding distance',
  confidence: 'SyncNet correlation peak confidence',
  region_weight_max: 'LipFD max temporal region weight',
}

export function unitLabels(unit?: DetectorUnit): { singular: string; plural: string; title: string } {
  switch (unit) {
    case 'window':
      return { singular: 'window', plural: 'windows', title: 'Window' }
    case 'frame':
      return { singular: 'frame', plural: 'frames', title: 'Frame' }
    case 'clip':
      return { singular: 'clip', plural: 'clips', title: 'Clip' }
    case 'face':
    default:
      return { singular: 'face', plural: 'faces', title: 'Face' }
  }
}

function bboxText(bbox: PerFace['bbox']): string {
  if (Array.isArray(bbox)) return bbox.join(', ')
  return bbox ? String(bbox) : '—'
}

function topRegionForFace(face: PerFace): string | null {
  const re = face.region_energy
  if (!re || typeof re !== 'object') return null
  let bestKey: string | null = null
  let bestVal = -1
  for (const [k, v] of Object.entries(re)) {
    const score =
      typeof v === 'number'
        ? v
        : typeof v === 'object' && v !== null
          ? Number(v.residual_pct ?? v.attention_pct ?? -1)
          : -1
    if (score > bestVal) {
      bestVal = score
      bestKey = k
    }
  }
  return bestKey ? `${bestKey.replace(/_/g, ' ')} (${bestVal.toFixed(1)}%)` : null
}

function rowAuxSummary(face: PerFace): string | null {
  const pairs: string[] = []
  if (face.info && typeof face.info === 'object') {
    for (const [k, v] of Object.entries(face.info)) {
      if (typeof v === 'number') {
        pairs.push(`${k}=${Number.isInteger(v) ? v : v.toFixed(3)}`)
      }
    }
  }
  if (face.aux && typeof face.aux === 'object') {
    for (const [k, v] of Object.entries(face.aux)) {
      if (typeof v === 'number') {
        pairs.push(`${k}=${v.toFixed(3)}`)
      }
    }
  }
  return pairs.length ? pairs.join(' · ') : null
}

export function PerFaceTable({
  rows,
  unit = 'face',
  inputMode,
}: {
  rows: PerFace[]
  unit?: DetectorUnit
  inputMode?: DetectorInputMode
}) {
  const [limit, setLimit] = useState(PER_FACE_PAGE)
  const labels = unitLabels(unit)
  const isWindow = unit === 'window' || inputMode === 'audio' || inputMode === 'av_clip'
  const isWholeFrame = unit === 'frame' || unit === 'clip'
  const hasRegionCol = !isWindow && !isWholeFrame && rows.some((r) => Boolean(r.region_energy))
  const hasAuxCol = rows.some((r) => Boolean(r.info || r.aux))

  const sorted = useMemo(
    () =>
      [...rows].sort(
        (a, b) =>
          (b.fake_prob ?? 0) - (a.fake_prob ?? 0) ||
          String(a.face_id ?? '').localeCompare(String(b.face_id ?? '')),
      ),
    [rows],
  )

  if (!rows.length) {
    return (
      <Text c="dimmed" fz="sm">
        This detector did not emit per-{labels.singular} scores.
      </Text>
    )
  }

  return (
    <>
      <Group justify="space-between" align="baseline" mb="xs">
        <div>
          <Text fz="sm" fw={600}>
            {rows.length} {rows.length === 1 ? labels.singular : labels.plural} evaluated
          </Text>
          <Text fz="xs" c="dimmed">
            Sorted by fake probability, highest first — matching the exported dossier.
          </Text>
        </div>
      </Group>
      <Table.ScrollContainer minWidth={560}>
        <Table striped highlightOnHover>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>{labels.title} ID</Table.Th>
              {isWindow ? (
                <Table.Th>Time span (s)</Table.Th>
              ) : (
                <>
                  <Table.Th>Frame</Table.Th>
                  <Table.Th>{isWholeFrame ? 'Dimensions (w × h)' : 'BBox (x, y, w, h)'}</Table.Th>
                </>
              )}
              {hasRegionCol ? <Table.Th>Peak attention region</Table.Th> : null}
              {hasAuxCol ? <Table.Th>Diagnostic telemetry</Table.Th> : null}
              <Table.Th w="16rem">Fake probability</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.slice(0, limit).map((face, index) => {
              const startS =
                face.start_s ??
                (Array.isArray(face.bbox) && typeof face.bbox[0] === 'number' ? face.bbox[0] : null)
              const endS =
                face.end_s ??
                (Array.isArray(face.bbox) && typeof face.bbox[1] === 'number' ? face.bbox[1] : null)
              const dims =
                Array.isArray(face.bbox) && face.bbox.length >= 4
                  ? `${face.bbox[2]} × ${face.bbox[3]}`
                  : bboxText(face.bbox)
              return (
                <Table.Tr key={`${face.face_id ?? index}`}>
                  <Table.Td ff="monospace">{face.face_id ?? '—'}</Table.Td>
                  {isWindow ? (
                    <Table.Td ff="monospace">
                      {startS != null && endS != null
                        ? `${Number(startS).toFixed(2)}s – ${Number(endS).toFixed(2)}s`
                        : '—'}
                    </Table.Td>
                  ) : (
                    <>
                      <Table.Td ff="monospace">{face.frame_idx ?? '—'}</Table.Td>
                      <Table.Td ff="monospace">{isWholeFrame ? dims : bboxText(face.bbox)}</Table.Td>
                    </>
                  )}
                  {hasRegionCol ? (
                    <Table.Td>
                      <Text fz="xs" c="dimmed">
                        {topRegionForFace(face) ?? '—'}
                      </Text>
                    </Table.Td>
                  ) : null}
                  {hasAuxCol ? (
                    <Table.Td>
                      <Text fz="xs" ff="monospace" c="dimmed">
                        {rowAuxSummary(face) ?? '—'}
                      </Text>
                    </Table.Td>
                  ) : null}
                  <Table.Td>
                    <ProbabilityBar value={face.fake_prob ?? 0} />
                  </Table.Td>
                </Table.Tr>
              )
            })}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      {limit < sorted.length ? (
        <Button
          variant="subtle"
          size="compact-sm"
          mt="xs"
          onClick={() => setLimit((value) => value + PER_FACE_PAGE * 4)}
        >
          Show more ({sorted.length - limit} remaining)
        </Button>
      ) : null}
    </>
  )
}

export function AuxSignalsTable({ finding }: { finding: Finding }) {
  const entries = Object.entries(finding.aux_aggregate ?? {}).filter(
    ([, val]) => typeof val === 'number' && !Number.isNaN(val),
  )
  if (!entries.length) return null

  return (
    <div>
      <Text fz="sm" fw={600} mb={2}>
        Auxiliary branch & sub-signal breakdown
      </Text>
      <Text fz="xs" c="dimmed" mb="xs" maw="48rem">
        Internal sub-branch scores and physical alignment metrics aggregated across all evaluated
        units.
      </Text>
      <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="xs">
        {entries.map(([key, value]) => {
          const isProb =
            key.includes('branch') ||
            key === 'deepfake' ||
            key === 'blend' ||
            key === 'identity_inconsistency'
          return (
            <Paper withBorder p="xs" key={key}>
              <Group justify="space-between" align="center" mb={isProb ? 4 : 0}>
                <Text fz="xs" c="dimmed" fw={500}>
                  {AUX_LABELS[key] ?? key.replace(/_/g, ' ')}
                </Text>
                <Badge variant="light" color={isProb && value >= 0.5 ? 'red' : 'indigo'} ff="monospace">
                  {value.toFixed(4)}
                </Badge>
              </Group>
              {isProb ? <ProbabilityBar value={value} showValue={false} /> : null}
            </Paper>
          )
        })}
      </SimpleGrid>
    </div>
  )
}

export function RegionEnergyTable({ finding }: { finding: Finding }) {
  const entries = Object.entries(finding.region_aggregate ?? {})
  if (!entries.length) return null

  const hasResidual = entries.some(([, value]) => value.residual_pct != null)
  const sorted = [...entries].sort(
    (a, b) =>
      (b[1].residual_pct ?? b[1].attention_pct ?? 0) -
      (a[1].residual_pct ?? a[1].attention_pct ?? 0),
  )

  return (
    <div>
      <Text fz="sm" fw={600}>
        Facial region energy attribution (7-region FaceMesh topology)
      </Text>
      <Text fz="xs" c="dimmed" mb="xs" maw="48rem">
        Proportion of spatial activation / residual energy falling within each anatomical landmark
        hull. Compare the highest-energy region against the attention maps.
      </Text>
      <Table.ScrollContainer minWidth={480}>
        <Table striped>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Facial region</Table.Th>
              {hasResidual ? <Table.Th w="13rem">Residual energy</Table.Th> : null}
              <Table.Th w="13rem">Attention energy</Table.Th>
              <Table.Th>Faces</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.map(([region, value]) => {
              const attn = value.attention_pct ?? 0
              const res = value.residual_pct ?? 0
              return (
                <Table.Tr key={region}>
                  <Table.Td fw={500} style={{ textTransform: 'capitalize' }}>
                    {region.replace(/_/g, ' ')}
                  </Table.Td>
                  {hasResidual ? (
                    <Table.Td>
                      <Group gap="xs" wrap="nowrap">
                        <Progress
                          value={Math.min(100, res)}
                          color="orange"
                          size="sm"
                          style={{ flex: 1 }}
                        />
                        <Text ff="monospace" fz="xs" w="3.2rem" ta="right">
                          {value.residual_pct != null ? `${value.residual_pct}%` : '—'}
                        </Text>
                      </Group>
                    </Table.Td>
                  ) : null}
                  <Table.Td>
                    <Group gap="xs" wrap="nowrap">
                      <Progress
                        value={Math.min(100, attn)}
                        color="indigo"
                        size="sm"
                        style={{ flex: 1 }}
                      />
                      <Text ff="monospace" fz="xs" w="3.2rem" ta="right">
                        {attn}%
                      </Text>
                    </Group>
                  </Table.Td>
                  <Table.Td ff="monospace">{value.face_count ?? 0}</Table.Td>
                </Table.Tr>
              )
            })}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </div>
  )
}

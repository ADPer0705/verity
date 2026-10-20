/** Tabular detail shared by the Findings and Evidence tabs. */

import { useMemo, useState } from 'react'
import { Button, Table, Text } from '@mantine/core'

import type { Finding, PerFace } from '@/api/types'
import { ProbabilityBar } from '@/components/Primitives'

const PER_FACE_PAGE = 40

function bboxText(bbox: PerFace['bbox']): string {
  if (Array.isArray(bbox)) return bbox.join(', ')
  return bbox ? String(bbox) : '—'
}

export function PerFaceTable({ rows }: { rows: PerFace[] }) {
  const [limit, setLimit] = useState(PER_FACE_PAGE)

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
        This detector did not emit per-face scores.
      </Text>
    )
  }

  return (
    <>
      <Text fz="sm" fw={500}>
        {rows.length} face{rows.length === 1 ? '' : 's'}
      </Text>
      <Text fz="xs" c="dimmed" mb="xs">
        Sorted by fake probability, highest first — the same ordering the report uses.
      </Text>
      <Table.ScrollContainer minWidth={520}>
        <Table striped>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Face id</Table.Th>
              <Table.Th>Frame</Table.Th>
              <Table.Th>BBox (x, y, w, h)</Table.Th>
              <Table.Th w="16rem">Fake probability</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.slice(0, limit).map((face, index) => (
              <Table.Tr key={`${face.face_id ?? index}`}>
                <Table.Td ff="monospace">{face.face_id ?? '—'}</Table.Td>
                <Table.Td>{face.frame_idx ?? '—'}</Table.Td>
                <Table.Td ff="monospace">{bboxText(face.bbox)}</Table.Td>
                <Table.Td>
                  <ProbabilityBar value={face.fake_prob ?? 0} />
                </Table.Td>
              </Table.Tr>
            ))}
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

export function RegionEnergyTable({ finding }: { finding: Finding }) {
  const entries = Object.entries(finding.region_aggregate ?? {})
  if (!entries.length) return null

  const hasResidual = entries.some(([, value]) => value.residual_pct != null)
  const sorted = entries.sort(
    (a, b) =>
      (b[1].residual_pct ?? b[1].attention_pct ?? 0) -
      (a[1].residual_pct ?? a[1].attention_pct ?? 0),
  )

  return (
    <div>
      <Text fz="sm" fw={500}>
        Face region energy attribution
      </Text>
      <Text fz="xs" c="dimmed" mb="xs" maw="48rem">
        Where in the face the model concentrated. High energy in a region that should be
        structurally uniform is the signal worth checking against the attention maps.
      </Text>
      <Table.ScrollContainer minWidth={420}>
        <Table striped>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Region</Table.Th>
              {hasResidual ? <Table.Th>Residual energy</Table.Th> : null}
              <Table.Th>Attention energy</Table.Th>
              <Table.Th>Faces</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.map(([region, value]) => (
              <Table.Tr key={region}>
                <Table.Td>{region.replace(/_/g, ' ')}</Table.Td>
                {hasResidual ? <Table.Td ff="monospace">{value.residual_pct ?? '—'}%</Table.Td> : null}
                <Table.Td ff="monospace">{value.attention_pct ?? 0}%</Table.Td>
                <Table.Td>{value.face_count ?? 0}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </div>
  )
}

/**
 * Detector selection — what ran, what did not, and why.
 *
 * This is the tab that makes the examination auditable. The CLI prints the
 * exclusion count; the report hides the reasons behind a disclosure. Here
 * they are the content: an investigator asked "why wasn't X used?" needs to
 * answer from the record, not from memory.
 */

import { useState } from 'react'
import { Badge, Group, Paper, SegmentedControl, Table, Text } from '@mantine/core'
import { IconFilter } from '@tabler/icons-react'

import type { CaseDetail } from '@/api/types'
import { EmptyState, SectionTitle } from '@/components/Primitives'

type Filterish = 'all' | 'included' | 'excluded'

export function SelectionTab({ detail }: { detail: CaseDetail }) {
  const [filter, setFilter] = useState<Filterish>('all')

  if (!detail.selection) {
    return (
      <EmptyState
        icon={<IconFilter size={26} />}
        title="No detector selection recorded"
        body="Run Analyze or a full examination. The selector evaluates every registered detector against this media and records a reason for each decision."
      />
    )
  }

  const decisions = detail.selection.decisions
  const included = decisions.filter((decision) => decision.included)
  const excluded = decisions.filter((decision) => !decision.included)
  const shown = filter === 'included' ? included : filter === 'excluded' ? excluded : decisions

  const ranIds = new Set(detail.findings.map((finding) => finding.detector_id))

  return (
    <div>
      <Group justify="space-between" align="flex-end" mb="sm" wrap="wrap">
        <div>
          <SectionTitle>Eligibility decisions</SectionTitle>
          <Text fz="sm" c="dimmed">
            {included.length} selected · {excluded.length} excluded · evaluated against media type,
            face presence and scale, clip length and weight availability.
          </Text>
        </div>
        <SegmentedControl
          size="xs"
          value={filter}
          onChange={(value) => setFilter(value as Filterish)}
          data={[
            { value: 'all', label: `All (${decisions.length})` },
            { value: 'included', label: `Selected (${included.length})` },
            { value: 'excluded', label: `Excluded (${excluded.length})` },
          ]}
        />
      </Group>

      <Paper withBorder>
        <Table.ScrollContainer minWidth={640}>
          <Table>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Method</Table.Th>
                <Table.Th>Decision</Table.Th>
                <Table.Th>Reasoning</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {shown.map((decision) => (
                <Table.Tr key={decision.detector_id}>
                  <Table.Td style={{ whiteSpace: 'nowrap' }}>
                    <Text fz="sm">{decision.detector_name}</Text>
                    <Text fz="xs" c="dimmed" ff="monospace">
                      {decision.detector_id}
                    </Text>
                  </Table.Td>
                  <Table.Td style={{ whiteSpace: 'nowrap' }}>
                    {decision.included ? (
                      <Badge color="teal" size="sm">
                        {ranIds.has(decision.detector_id) ? 'Ran' : 'Selected — no finding'}
                      </Badge>
                    ) : (
                      <Badge color="gray" size="sm">
                        Excluded
                      </Badge>
                    )}
                  </Table.Td>
                  <Table.Td>
                    <ul className="vy-reason-list">
                      {decision.reasons.map((reason, index) => (
                        <li key={`${decision.detector_id}-${index}`}>{reason}</li>
                      ))}
                    </ul>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        {shown.length === 0 ? (
          <Text c="dimmed" fz="sm" p="md">
            Nothing in this category.
          </Text>
        ) : null}
      </Paper>

      <Text fz="xs" c="dimmed" mt="sm" maw="48rem">
        "Selected — no finding" means the detector was eligible but produced no result: it either
        failed during the run or has not been executed yet. Check the Activity tab for the
        detector's stderr.
      </Text>
    </div>
  )
}

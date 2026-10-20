/**
 * Job history and live output.
 *
 * Used both as a standalone page (all jobs) and embedded in a case (that
 * case's jobs). Jobs are held in memory by the server, so this is a session
 * record, not an audit log — the durable record is what each stage wrote to
 * the workspace.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Anchor, Badge, Paper, Stack, Table, Text } from '@mantine/core'
import { IconActivity } from '@tabler/icons-react'

import { api } from '@/api/client'
import { useJobStream, usePolling } from '@/api/hooks'
import { JobConsole, STATUS_COLOR } from '@/components/JobConsole'
import { EmptyState, ErrorBanner, PageHeader } from '@/components/Primitives'
import { formatDurationBetween, formatRelative } from '@/lib/format'

export function ActivityPanel({ mediaId }: { mediaId?: string }) {
  const [selected, setSelected] = useState<string | null>(null)
  const { data, error, reload } = usePolling(
    () => api.jobs({ mediaId, limit: 60 }),
    4000,
    true,
    [mediaId],
  )
  const stream = useJobStream(selected)

  const jobs = data?.jobs ?? []

  return (
    <Stack gap="md">
      <ErrorBanner error={error} onRetry={reload} />

      {jobs.length === 0 ? (
        <EmptyState
          icon={<IconActivity size={26} />}
          title={mediaId ? 'No jobs run for this case yet' : 'No jobs in this session'}
          body="Pipeline operations started from this interface appear here with their stages and full output."
        />
      ) : (
        <Paper withBorder>
          <div style={{ padding: '0.75rem 0.75rem 0.5rem' }}>
            <Text fw={500} fz="sm">
              {mediaId ? 'Jobs for this case' : 'Recent jobs'}
            </Text>
            <Text fz="xs" c="dimmed">
              Select a job to see its stages and output.
            </Text>
          </div>
          <Table.ScrollContainer minWidth={640}>
            <Table highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Operation</Table.Th>
                  <Table.Th>Status</Table.Th>
                  {mediaId ? null : <Table.Th>Case</Table.Th>}
                  <Table.Th>Started</Table.Th>
                  <Table.Th>Duration</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {jobs.map((job) => (
                  <Table.Tr
                    key={job.id}
                    className="vy-clickable"
                    onClick={() => setSelected(job.id)}
                    bg={selected === job.id ? 'var(--mantine-color-default-hover)' : undefined}
                  >
                    <Table.Td>
                      <Text fz="sm">{job.label}</Text>
                      <Text fz="xs" c="dimmed" ff="monospace">
                        {job.kind} · {job.id}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Badge color={STATUS_COLOR[job.status]} size="sm">
                        {job.status}
                      </Badge>
                    </Table.Td>
                    {mediaId ? null : (
                      <Table.Td>
                        {job.media_ids.length ? (
                          <Anchor
                            component={Link}
                            to={`/case/${job.media_ids[0]}`}
                            ff="monospace"
                            fz="sm"
                            onClick={(event) => event.stopPropagation()}
                          >
                            {job.media_ids[0]}
                          </Anchor>
                        ) : (
                          <Text c="dimmed">—</Text>
                        )}
                      </Table.Td>
                    )}
                    <Table.Td>{formatRelative(job.started_at ?? job.created_at)}</Table.Td>
                    <Table.Td ff="monospace">
                      {formatDurationBetween(job.started_at ?? job.created_at, job.finished_at)}
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </Paper>
      )}

      {selected ? (
        <JobConsole
          job={stream.job}
          logs={stream.logs}
          onCancel={() => api.cancelJob(selected).catch(() => undefined)}
        />
      ) : null}
    </Stack>
  )
}

export default function ActivityPage() {
  return (
    <div className="vy-page">
      <PageHeader
        title="Activity"
        subtitle="Pipeline operations started in this server session, with their stages and full output."
      />
      <ActivityPanel />
    </div>
  )
}

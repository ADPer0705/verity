/**
 * Live job progress.
 *
 * Two levels of detail from one stream: `variant="steps"` shows only the
 * named stages (the at-a-glance progress view), `variant="full"` adds the
 * raw log an examiner has to be able to quote.
 */

import { useEffect, useRef } from 'react'
import { Badge, Button, Group, Loader, Paper, Text, Timeline } from '@mantine/core'
import { IconCheck, IconMinus, IconX } from '@tabler/icons-react'

import type { Job, JobLogLine, JobStep } from '@/api/types'
import { formatDurationBetween } from '@/lib/format'

export const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'gray',
  running: 'indigo',
  succeeded: 'teal',
  failed: 'red',
  cancelled: 'pink',
}

function bulletFor(step: JobStep) {
  switch (step.status) {
    case 'succeeded':
      return <IconCheck size={12} />
    case 'failed':
      return <IconX size={12} />
    case 'running':
      return <Loader size={10} color="white" />
    case 'skipped':
      return <IconMinus size={12} />
    default:
      return null
  }
}

const STEP_COLOR: Record<JobStep['status'], string> = {
  pending: 'gray',
  running: 'indigo',
  succeeded: 'teal',
  failed: 'red',
  skipped: 'gray',
}

export function JobSteps({ steps }: { steps: JobStep[] }) {
  if (!steps.length) return null
  // Everything up to the last step that has started is "active" for the line.
  let active = -1
  steps.forEach((step, index) => {
    if (step.status !== 'pending') active = index
  })

  return (
    <Timeline active={active} bulletSize={20} lineWidth={2} p="md">
      {steps.map((step) => (
        <Timeline.Item
          key={step.key}
          bullet={bulletFor(step)}
          color={STEP_COLOR[step.status]}
          lineVariant={step.status === 'pending' ? 'dashed' : 'solid'}
          title={
            <Text fz="sm" fw={500} c={step.status === 'pending' ? 'dimmed' : undefined}>
              {step.label}
            </Text>
          }
        >
          {step.detail ? (
            <Text fz="xs" c="dimmed" className="vy-break">
              {step.detail}
            </Text>
          ) : null}
        </Timeline.Item>
      ))}
    </Timeline>
  )
}

export function JobLog({ logs }: { logs: JobLogLine[] }) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'nearest' })
  }, [logs.length])

  if (!logs.length) {
    return (
      <div className="vy-log">
        <Text fz="xs" c="dimmed" ff="monospace">
          Waiting for the first output from the pipeline…
        </Text>
      </div>
    )
  }

  return (
    <div className="vy-log" role="log" aria-live="polite">
      {logs.map((line) => (
        <div key={line.seq} className={`vy-log__line vy-log__line--${line.level}`}>
          <span className="vy-log__ts">{new Date(line.ts).toLocaleTimeString()}</span>
          {line.message}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  )
}

export function JobConsole({
  job,
  logs,
  variant = 'full',
  onCancel,
}: {
  job: Job | null
  logs: JobLogLine[]
  variant?: 'steps' | 'full'
  onCancel?: () => void
}) {
  if (!job) {
    return (
      <Paper withBorder p="md">
        <Group gap="xs">
          <Loader size="xs" />
          <Text fz="sm" c="dimmed">
            Starting…
          </Text>
        </Group>
      </Paper>
    )
  }

  const active = job.status === 'running' || job.status === 'pending'

  return (
    <Paper withBorder>
      <Group justify="space-between" px="md" pt="sm" wrap="wrap">
        <Group gap="xs">
          <Badge color={STATUS_COLOR[job.status]} variant="filled">
            {job.status}
          </Badge>
          <Text fz="sm" c="dimmed">
            {job.label}
          </Text>
        </Group>
        <Group gap="xs">
          <Text fz="xs" c="dimmed" ff="monospace">
            {formatDurationBetween(job.started_at ?? job.created_at, job.finished_at)}
          </Text>
          {active && onCancel ? (
            <Button variant="subtle" color="red" size="compact-sm" onClick={onCancel}>
              Cancel
            </Button>
          ) : null}
        </Group>
      </Group>

      <JobSteps steps={job.steps} />

      {job.error ? (
        <div className="vy-log vy-log__line--error" style={{ maxHeight: '8rem' }}>
          {job.error}
        </div>
      ) : null}

      {variant === 'full' ? <JobLog logs={logs} /> : null}
    </Paper>
  )
}

/** Small shared building blocks used by both profiles. */

import type { ReactNode } from 'react'
import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  CopyButton,
  Group,
  Notification,
  Paper,
  Progress,
  SimpleGrid,
  Skeleton,
  Stack,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
} from '@mantine/core'
import { IconAlertCircle, IconCheck, IconCopy } from '@tabler/icons-react'

import type { CoverageFamily, DetectorTier } from '@/api/types'
import { familyLabel } from '@/lib/taxonomy'

// --------------------------------------------------------------------- //

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
}) {
  return (
    <Group justify="space-between" align="flex-start" mb="lg" wrap="wrap">
      <div>
        <Title order={1}>{title}</Title>
        {subtitle ? (
          <Text c="dimmed" fz="sm" mt={4}>
            {subtitle}
          </Text>
        ) : null}
      </div>
      {actions ? <Group gap="xs">{actions}</Group> : null}
    </Group>
  )
}

export function SectionTitle({ children, mt }: { children: ReactNode; mt?: string | number }) {
  return (
    <h3 className="vy-section-title" style={mt ? { marginTop: mt } : undefined}>
      {children}
    </h3>
  )
}

// --------------------------------------------------------------------- //

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon?: ReactNode
  title: string
  body?: ReactNode
  action?: ReactNode
}) {
  return (
    <Stack align="center" gap="sm" py="xl" px="md" ta="center">
      {icon ? (
        <ThemeIcon variant="light" color="gray" size={48} radius="md">
          {icon}
        </ThemeIcon>
      ) : null}
      <Title order={3}>{title}</Title>
      {body ? (
        <Text c="dimmed" fz="sm" maw="38rem" lh={1.55}>
          {body}
        </Text>
      ) : null}
      {action ? <div style={{ marginTop: '0.5rem' }}>{action}</div> : null}
    </Stack>
  )
}

// --------------------------------------------------------------------- //

export function ErrorBanner({
  error,
  title = 'Something went wrong',
  onRetry,
}: {
  error: string | null
  title?: string
  onRetry?: () => void
}) {
  if (!error) return null
  return (
    <Notification
      color="red"
      title={title}
      withCloseButton={false}
      withBorder
      mb="md"
      icon={<IconAlertCircle size={16} />}
    >
      <Text fz="sm" className="vy-break">
        {error}
      </Text>
      {onRetry ? (
        <Button size="compact-xs" variant="light" color="red" mt="xs" onClick={onRetry}>
          Retry
        </Button>
      ) : null}
    </Notification>
  )
}

/** Inline explanatory note — info, warning, success or error. */
export function Banner({
  color,
  title,
  children,
  icon,
  mt,
  mb,
}: {
  color: 'blue' | 'yellow' | 'teal' | 'red' | 'gray'
  title?: ReactNode
  children?: ReactNode
  icon?: ReactNode
  mt?: string | number
  mb?: string | number
}) {
  return (
    <Alert variant="light" color={color} title={title} icon={icon} mt={mt} mb={mb}>
      {children}
    </Alert>
  )
}

// --------------------------------------------------------------------- //

export function LoadingBlock({ lines = 3 }: { lines?: number }) {
  return (
    <Stack gap="xs" aria-busy>
      <Skeleton height={44} />
      {Array.from({ length: lines }, (_, index) => (
        <Skeleton key={index} height={14} width={`${90 - (index % 3) * 12}%`} />
      ))}
    </Stack>
  )
}

// --------------------------------------------------------------------- //

/** Monospace value with a copy affordance — checksums, ids, paths. */
export function CopyValue({
  value,
  label,
  truncate,
}: {
  value: string
  label?: string
  truncate?: string
}) {
  return (
    <Group gap={6} wrap="nowrap" align="center" display="inline-flex">
      <Text component="span" ff="monospace" fz="sm" className="vy-break" title={value}>
        {truncate ?? value}
      </Text>
      <CopyButton value={value} timeout={1600}>
        {({ copied, copy }) => (
          <Tooltip label={copied ? 'Copied' : `Copy ${label ?? 'value'}`} position="top">
            <ActionIcon
              variant="subtle"
              color={copied ? 'teal' : 'gray'}
              size="sm"
              aria-label={`Copy ${label ?? 'value'}`}
              onClick={copy}
            >
              {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
            </ActionIcon>
          </Tooltip>
        )}
      </CopyButton>
    </Group>
  )
}

// --------------------------------------------------------------------- //

/** Horizontal bar for a 0–1 probability, coloured by band. */
export function ProbabilityBar({ value, showValue = true }: { value: number; showValue?: boolean }) {
  const clamped = Math.max(0, Math.min(1, value))
  const color = clamped >= 0.7 ? 'red' : clamped >= 0.3 ? 'orange' : 'teal'
  return (
    <Group gap="xs" wrap="nowrap" align="center">
      <Progress
        value={clamped * 100}
        color={color}
        size="sm"
        style={{ flex: 1, minWidth: '5rem' }}
        aria-label={`Mean fake probability ${clamped.toFixed(4)} of 1`}
      />
      {showValue ? (
        <Text component="span" ff="monospace" fz="xs" w="3.25rem" ta="right">
          {clamped.toFixed(4)}
        </Text>
      ) : null}
    </Group>
  )
}

// --------------------------------------------------------------------- //

export interface Metric {
  label: string
  value: ReactNode
  hint?: ReactNode
}

export function MetricGrid({ metrics }: { metrics: Metric[] }) {
  return (
    <SimpleGrid cols={{ base: 2, md: 4 }} spacing="xs">
      {metrics.map((metric) => (
        <Paper withBorder p="sm" key={metric.label}>
          <Text fz="xs" c="dimmed" fw={600} tt="uppercase" style={{ letterSpacing: '0.05em' }}>
            {metric.label}
          </Text>
          <Text fz="xl" fw={500} mt={2} lh={1.2}>
            {metric.value}
          </Text>
          {metric.hint ? (
            <Text fz="xs" c="dimmed" mt={4}>
              {metric.hint}
            </Text>
          ) : null}
        </Paper>
      ))}
    </SimpleGrid>
  )
}

// --------------------------------------------------------------------- //

export function KeyValue({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="vy-kv">
      {rows.map(([key, value]) => (
        <div key={key} style={{ display: 'contents' }}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

// --------------------------------------------------------------------- //

/** Classification head tier. Anything but "trained" is red: it is not evidence. */
export function TierBadge({ tier, size = 'sm' }: { tier: DetectorTier; size?: 'xs' | 'sm' }) {
  if (tier === 'trained') {
    return (
      <Badge
        color="teal"
        size={size}
        style={{ flexShrink: 0, overflow: 'visible' }}
        title="Released deepfake-trained checkpoint or training-free method"
      >
        trained
      </Badge>
    )
  }
  return (
    <Badge
      color="red"
      variant="filled"
      size={size}
      title="No deepfake-trained classification head: outputs are noise, research use only"
    >
      {tier}
    </Badge>
  )
}

/** Manipulation families a detector targets, as small chips. */
export function FamilyChips({
  targets,
  families,
  size = 'xs',
}: {
  targets: string[] | null | undefined
  families?: CoverageFamily[] | null
  size?: 'xs' | 'sm'
}) {
  if (!targets?.length) {
    return (
      <Text component="span" c="dimmed" fz="xs">
        —
      </Text>
    )
  }
  return (
    <Group gap={4}>
      {targets.map((target) => (
        <Badge key={target} size={size} variant="outline" color="gray" title={target}>
          {familyLabel(target, families)}
        </Badge>
      ))}
    </Group>
  )
}

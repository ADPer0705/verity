/**
 * Detector registry, manipulation-family coverage and provisioning
 * (`verity setup`).
 *
 * Coverage comes first: which kinds of manipulation this installation can
 * actually examine with evidence-grade methods, and which it cannot. Weight
 * availability is the thing that most often silently removes a detector from
 * an examination, so it stays a prominent column rather than a footnote.
 */

import { Fragment, useMemo, useState } from 'react'
import {
  ActionIcon,
  Anchor,
  Badge,
  Button,
  Group,
  Paper,
  SegmentedControl,
  Select,
  Table,
  Text,
  TextInput,
} from '@mantine/core'
import {
  IconChevronDown,
  IconChevronRight,
  IconCloudDownload,
  IconPackage,
  IconRefresh,
  IconSearch,
} from '@tabler/icons-react'

import { ApiError, api } from '@/api/client'
import { useAsync, useJobStream } from '@/api/hooks'
import type { CoverageFamily, DetectorEntry } from '@/api/types'
import { JobConsole } from '@/components/JobConsole'
import {
  Banner,
  EmptyState,
  ErrorBanner,
  FamilyChips,
  KeyValue,
  LoadingBlock,
  PageHeader,
  SectionTitle,
  TierBadge,
} from '@/components/Primitives'
import { formatBytes } from '@/lib/format'
import { FAMILIES } from '@/lib/taxonomy'

type TierFilter = 'all' | 'trained' | 'untrained'

function WeightsBadge({ detector }: { detector: DetectorEntry }) {
  const { weights } = detector
  if (weights.available) {
    return (
      <Badge color="teal" size="sm">
        {weights.self_contained ? 'self-contained' : 'weights present'}
      </Badge>
    )
  }
  return (
    <Badge color="red" size="sm">
      checkpoint missing
    </Badge>
  )
}

function FamilyCard({
  family,
  selected,
  onSelect,
}: {
  family: CoverageFamily
  selected: boolean
  onSelect: () => void
}) {
  const state = family.covered ? 'covered' : family.untrained.length ? 'partial' : 'none'
  return (
    <div
      className={`vy-family vy-family--${state} vy-clickable`}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSelect()
        }
      }}
      style={selected ? { outline: '2px solid var(--mantine-primary-color-filled)' } : undefined}
    >
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="xs">
        <Text fw={500} fz="sm">
          {family.label}
        </Text>
        {family.covered ? (
          <Badge color="teal" size="sm">
            {family.trained.length} evidence-grade
          </Badge>
        ) : family.untrained.length ? (
          <Badge color="orange" size="sm">
            research only
          </Badge>
        ) : (
          <Badge color="gray" variant="outline" size="sm">
            not covered
          </Badge>
        )}
      </Group>
      <p className="vy-family__desc">{family.description}</p>
      <Group gap={4} mb={6}>
        {family.modalities.map((modality) => (
          <Badge key={modality} size="xs" variant="outline" color="gray">
            {modality}
          </Badge>
        ))}
      </Group>
      {family.trained.length ? (
        <Text fz="xs" ff="monospace" className="vy-break">
          {family.trained.join(', ')}
        </Text>
      ) : (
        <Text fz="xs" c="dimmed">
          No evidence-grade detector targets this family.
        </Text>
      )}
      {family.untrained.length ? (
        <Text fz="xs" c="dimmed" mt={2}>
          + {family.untrained.length} research-only detector
          {family.untrained.length === 1 ? '' : 's'} (not evidence)
        </Text>
      ) : null}
    </div>
  )
}

export default function Detectors() {
  const catalog = useAsync(() => api.detectors(), [])
  const coverage = useAsync(() => api.coverage(), [])
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState<string | null>(null)
  const [tier, setTier] = useState<TierFilter>('trained')
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const stream = useJobStream(jobId)

  const families = coverage.data?.families ?? FAMILIES
  const covered = coverage.data?.covered ?? families.filter((f) => f.covered).length

  const provision = async (ids: string[] | null, force = false) => {
    setError(null)
    try {
      const job = await api.setupDetectors(ids, force)
      setJobId(job.id)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    }
  }

  const reload = () => {
    catalog.reload()
    coverage.reload()
  }

  const detectors = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return (catalog.data?.detectors ?? []).filter((detector) => {
      if (family && !(detector.targets ?? []).includes(family)) return false
      if (tier === 'trained' && !detector.trained) return false
      if (tier === 'untrained' && detector.trained) return false
      if (!needle) return true
      return (
        detector.detector_id.toLowerCase().includes(needle) ||
        detector.name.toLowerCase().includes(needle) ||
        detector.description.toLowerCase().includes(needle)
      )
    })
  }, [catalog.data, query, family, tier])

  const toggle = (id: string) =>
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const busy = jobId !== null && !stream.finished

  return (
    <div className="vy-page">
      <PageHeader
        title="Detectors"
        subtitle={
          catalog.data
            ? `${catalog.data.count} registered · ${catalog.data.weights_ready} ready to run · ${covered} of ${families.length} manipulation families covered by an evidence-grade method`
            : 'Reading the registry…'
        }
        actions={
          <>
            <Button variant="default" leftSection={<IconRefresh size={16} />} onClick={reload}>
              Rescan
            </Button>
            <Button
              leftSection={<IconPackage size={16} />}
              disabled={!catalog.data?.count || busy}
              onClick={() =>
                provision(
                  (catalog.data?.detectors ?? [])
                    .filter((detector) => detector.trained)
                    .map((detector) => detector.detector_id),
                )
              }
              title="Builds the Python environment of every evidence-grade detector. Several GB of downloads."
            >
              Install evidence-grade detectors
            </Button>
          </>
        }
      />

      <ErrorBanner error={catalog.error ?? error} onRetry={reload} />

      {jobId ? (
        <div style={{ marginBottom: '1rem' }}>
          <JobConsole
            job={stream.job}
            logs={stream.logs}
            onCancel={() => api.cancelJob(jobId).catch(() => undefined)}
          />
        </div>
      ) : null}

      {catalog.loading && !catalog.data ? (
        <LoadingBlock lines={6} />
      ) : !catalog.data?.detectors_dir_present ? (
        <EmptyState
          icon={<IconCloudDownload size={26} />}
          title="Detectors directory not found"
          body={
            <>
              Verity looked in{' '}
              <span className="vy-mono vy-break">{catalog.data?.detectors_dir}</span>. Point it
              somewhere else under Settings, or set the{' '}
              <span className="vy-mono">VERITY_DETECTORS_DIR</span> environment variable. Each
              detector is a directory containing a <span className="vy-mono">detector.yaml</span>{' '}
              definition.
            </>
          }
        />
      ) : catalog.data.count === 0 ? (
        <EmptyState
          icon={<IconCloudDownload size={26} />}
          title="No detector definitions found"
          body={
            <>
              <span className="vy-mono vy-break">{catalog.data.detectors_dir}</span> exists but
              contains no <span className="vy-mono">detector.yaml</span> files.
            </>
          }
        />
      ) : (
        <>
          {/* ---------------- coverage ---------------- */}
          <SectionTitle>Coverage by manipulation family</SectionTitle>
          <Text fz="sm" c="dimmed" mb="sm" maw="60rem">
            Each detector declares the manipulation families it was trained to recognise. A family
            counts as covered only when at least one evidence-grade (trained-head) detector targets
            it; research-only detectors are counted but their output is not evidence.
            Select a family to filter the table below.
          </Text>
          {coverage.error ? (
            <Banner color="yellow" title="Coverage unavailable" mb="sm">
              {coverage.error} — showing the taxonomy without detector assignments.
            </Banner>
          ) : null}
          <div className="vy-coverage">
            {families.map((item) => (
              <FamilyCard
                key={item.id}
                family={item}
                selected={family === item.id}
                onSelect={() => setFamily((current) => (current === item.id ? null : item.id))}
              />
            ))}
          </div>

          {/* ---------------- registry ---------------- */}
          <SectionTitle mt="2rem">Registry</SectionTitle>
          <Paper withBorder>
            <Group p="sm" gap="sm" wrap="wrap" align="flex-end">
              <TextInput
                aria-label="Search detectors"
                placeholder="Filter by name, id or description"
                leftSection={<IconSearch size={14} />}
                value={query}
                onChange={(event) => setQuery(event.currentTarget.value)}
                style={{ flex: '1 1 16rem' }}
              />
              <Select
                aria-label="Filter by manipulation family"
                placeholder="Any family"
                clearable
                searchable
                data={families.map((item) => ({ value: item.id, label: item.label }))}
                value={family}
                onChange={setFamily}
                w={{ base: '100%', sm: 220 }}
              />
              <SegmentedControl
                size="xs"
                value={tier}
                onChange={(value) => setTier(value as TierFilter)}
                data={[
                  { value: 'trained', label: 'Evidence-grade' },
                  { value: 'untrained', label: 'Research-only' },
                  { value: 'all', label: 'All' },
                ]}
              />
              <Text fz="xs" c="dimmed">
                {detectors.length} of {catalog.data.count}
              </Text>
            </Group>

            <Table.ScrollContainer minWidth={960}>
              <Table highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th w={36} aria-label="Expand" />
                    <Table.Th>Method</Table.Th>
                    <Table.Th>Accepts</Table.Th>
                    <Table.Th>Targets</Table.Th>
                    <Table.Th miw={110}>Head</Table.Th>
                    <Table.Th>Weights</Table.Th>
                    <Table.Th miw={130}>Explainability</Table.Th>
                    <Table.Th>Actions</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {detectors.map((detector) => {
                    const open = expanded.has(detector.detector_id)
                    return (
                      <Fragment key={detector.detector_id}>
                        <Table.Tr>
                          <Table.Td>
                            <ActionIcon
                              variant="subtle"
                              color="gray"
                              size="sm"
                              aria-label={`${open ? 'Collapse' : 'Expand'} ${detector.name}`}
                              aria-expanded={open}
                              onClick={() => toggle(detector.detector_id)}
                            >
                              {open ? <IconChevronDown size={16} /> : <IconChevronRight size={16} />}
                            </ActionIcon>
                          </Table.Td>
                          <Table.Td>
                            <Text fz="sm">{detector.name}</Text>
                            <Text fz="xs" c="dimmed" ff="monospace">
                              {detector.detector_id} v{detector.version}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            {detector.media_types.join(', ')}
                            {detector.faces_required ? (
                              <Text fz="xs" c="dimmed">
                                faces required, min {detector.min_face}px
                              </Text>
                            ) : null}
                          </Table.Td>
                          <Table.Td>
                            <FamilyChips targets={detector.targets} families={families} />
                          </Table.Td>
                          <Table.Td>
                            <TierBadge tier={detector.tier} />
                          </Table.Td>
                          <Table.Td>
                            <WeightsBadge detector={detector} />
                          </Table.Td>
                          <Table.Td>
                            {detector.explainability ? (
                              <Badge color="indigo" size="sm">
                                activation maps
                              </Badge>
                            ) : (
                              <Text c="dimmed">—</Text>
                            )}
                          </Table.Td>
                          <Table.Td>
                            <Button
                              variant="subtle"
                              size="compact-xs"
                              disabled={busy}
                              onClick={() => provision([detector.detector_id])}
                            >
                              Install
                            </Button>
                          </Table.Td>
                        </Table.Tr>
                        {open ? (
                          <Table.Tr>
                            <Table.Td colSpan={8} bg="var(--mantine-color-default-hover)">
                              <div style={{ padding: '0.5rem 0.25rem 0.75rem', maxWidth: '64rem' }}>
                                {detector.description ? (
                                  <Text fz="sm" mb="sm" lh={1.55}>
                                    {detector.description}
                                  </Text>
                                ) : null}
                                <KeyValue
                                  rows={[
                                    [
                                      'Targets',
                                      <FamilyChips
                                        key="t"
                                        targets={detector.targets}
                                        families={families}
                                        size="sm"
                                      />,
                                    ],
                                    ['Media types', detector.media_types.join(', ')],
                                    ['Faces required', detector.faces_required ? 'yes' : 'no'],
                                    ['Minimum face size', `${detector.min_face} px`],
                                    ['Input resolution', `${detector.resolution} px`],
                                    [
                                      'Temporal',
                                      detector.video_mode
                                        ? `video mode, clip size ${detector.clip_size ?? '—'}`
                                        : 'per-frame',
                                    ],
                                    ['Weight kind', detector.weights.kind],
                                    ['Classification head', detector.tier],
                                    ...(detector.weights.license
                                      ? ([['License', detector.weights.license]] as [string, string][])
                                      : []),
                                    ['Checkpoint file', detector.weights.file || '—'],
                                    [
                                      'Checkpoint size',
                                      detector.weights.size ? formatBytes(detector.weights.size) : '—',
                                    ],
                                    [
                                      'Source',
                                      detector.weights.url ? (
                                        <Anchor
                                          key="u"
                                          href={detector.weights.url}
                                          target="_blank"
                                          rel="noreferrer"
                                          className="vy-break"
                                          fz="sm"
                                        >
                                          {detector.weights.url}
                                        </Anchor>
                                      ) : (
                                        '—'
                                      ),
                                    ],
                                    ...(detector.weights.note
                                      ? ([['Note', detector.weights.note]] as [string, string][])
                                      : []),
                                  ]}
                                />
                              </div>
                            </Table.Td>
                          </Table.Tr>
                        ) : null}
                      </Fragment>
                    )
                  })}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
            {detectors.length === 0 ? (
              <Text c="dimmed" fz="sm" p="md">
                No detector matches the current filters.
              </Text>
            ) : null}
          </Paper>

          <Banner color="blue" title="About installing detectors" mt="md">
            Each detector runs in its own Python environment, built from that detector's own pins.
            Installing builds that environment. Model checkpoints marked missing above are fetched
            separately with <code>verity weights pull</code>. The container images ship with every
            evidence-grade environment already installed.
          </Banner>
        </>
      )}
    </div>
  )
}

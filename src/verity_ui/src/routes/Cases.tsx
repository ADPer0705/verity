/**
 * Home: examine a new file, or reopen a case.
 *
 * Every ingested media item is a case, listed with enough state to decide
 * what needs attention: whether it has been examined, how many detectors
 * ran, whether the panel agreed, and whether a report exists. Dropping a
 * file uploads it and opens the case with the run dialog ready, so the
 * detector set is always a deliberate choice.
 */

import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Badge,
  Button,
  Group,
  Paper,
  Progress,
  Table,
  Text,
  TextInput,
  UnstyledButton,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import {
  IconChevronDown,
  IconChevronUp,
  IconRefresh,
  IconSearch,
  IconSelector,
  IconUpload,
} from '@tabler/icons-react'

import { ApiError, api } from '@/api/client'
import { usePolling } from '@/api/hooks'
import type { CaseSummary } from '@/api/types'
import { Banner, ErrorBanner, LoadingBlock, PageHeader } from '@/components/Primitives'
import { formatBytes, formatRelative, shortHash } from '@/lib/format'
import { presentBand } from '@/lib/verdict'

type SortKey = 'file_name' | 'media_type' | 'assessment' | 'detectors' | 'size' | 'ingested' | 'report'

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'file_name', label: 'File' },
  { key: 'media_type', label: 'Type' },
  { key: 'assessment', label: 'Assessment' },
  { key: 'detectors', label: 'Detectors' },
  { key: 'size', label: 'Size' },
  { key: 'ingested', label: 'Ingested' },
  { key: 'report', label: 'Report' },
]

function sortValue(item: CaseSummary, key: SortKey): string | number {
  switch (key) {
    case 'file_name':
      return item.file_name.toLowerCase()
    case 'media_type':
      return item.media_type
    case 'assessment':
      return presentBand(item.consensus.band).label
    case 'detectors':
      return item.findings_count
    case 'size':
      return item.size_bytes
    case 'ingested':
      return item.ingested_at ?? ''
    case 'report':
      return item.reports.pdf ? 2 : item.reports.html ? 1 : 0
  }
}

function SortHeader({
  column,
  active,
  direction,
  onClick,
}: {
  column: { key: SortKey; label: string }
  active: boolean
  direction: 'asc' | 'desc'
  onClick: () => void
}) {
  const Icon = !active ? IconSelector : direction === 'asc' ? IconChevronUp : IconChevronDown
  return (
    <Table.Th aria-sort={active ? (direction === 'asc' ? 'ascending' : 'descending') : undefined}>
      <UnstyledButton onClick={onClick} fz="sm" fw={600}>
        <Group gap={4} wrap="nowrap">
          <span>{column.label}</span>
          <Icon size={14} stroke={1.5} />
        </Group>
      </UnstyledButton>
    </Table.Th>
  )
}

export default function Cases() {
  const navigate = useNavigate()
  const { data, error, loading, reload } = usePolling(() => api.cases(), 8000, true, [])
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<{ key: SortKey; direction: 'asc' | 'desc' }>({
    key: 'ingested',
    direction: 'desc',
  })

  const [file, setFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)
  const [busy, setBusy] = useState(false)
  const [intakeError, setIntakeError] = useState<string | null>(null)

  const cases = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const filtered = (data?.cases ?? []).filter(
      (item) =>
        !needle ||
        item.file_name.toLowerCase().includes(needle) ||
        item.media_id.toLowerCase().includes(needle) ||
        item.checksum.toLowerCase().includes(needle),
    )
    const sign = sort.direction === 'asc' ? 1 : -1
    return [...filtered].sort((a, b) => {
      const left = sortValue(a, sort.key)
      const right = sortValue(b, sort.key)
      if (left < right) return -1 * sign
      if (left > right) return 1 * sign
      return 0
    })
  }, [data, query, sort])

  const toggleSort = (key: SortKey) =>
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { key, direction: key === 'file_name' || key === 'media_type' ? 'asc' : 'desc' },
    )

  const ingest = async (chosen: File) => {
    setFile(chosen)
    setBusy(true)
    setProgress(0)
    setIntakeError(null)
    try {
      const uploaded = await api.upload(chosen, setProgress)
      navigate(`/case/${uploaded.media_id}?run=1`)
    } catch (cause) {
      setIntakeError(cause instanceof ApiError ? cause.message : String(cause))
      setBusy(false)
    }
  }

  return (
    <div className="vy-page">
      <PageHeader
        title="Examine a file"
        subtitle="Photo, video or audio. The file is copied read-only into the workspace and fingerprinted with SHA-256; nothing leaves this machine."
      />

      <Dropzone
        onDrop={(files) => files[0] && void ingest(files[0])}
        onReject={() => setIntakeError('Unsupported file type. Choose a video, image or audio file.')}
        accept={['video/*', 'image/*', 'audio/*']}
        multiple={false}
        disabled={busy}
        mb="md"
      >
        <Group justify="center" gap="md" mih={110} style={{ pointerEvents: 'none' }}>
          <IconUpload size={30} stroke={1.5} />
          <div>
            <Text fz="md" fw={500}>
              {busy ? `Uploading ${file?.name ?? ''}` : 'Drop a file here, or click to choose one'}
            </Text>
            <Text fz="sm" c="dimmed">
              Next you choose which detectors run. Re-uploading identical bytes reopens the same case.
            </Text>
          </div>
        </Group>
      </Dropzone>

      {busy ? (
        <div style={{ marginBottom: '1rem' }}>
          <Progress value={progress * 100} size="sm" animated={progress >= 1} />
          <Text fz="xs" c="dimmed" mt={4}>
            {progress < 1
              ? `Uploading ${formatBytes(file?.size ?? 0)} — ${Math.round(progress * 100)}%`
              : 'Fingerprinting and opening the case…'}
          </Text>
        </div>
      ) : null}

      {intakeError ? (
        <Banner color="red" title="Upload failed" mb="md">
          {intakeError}
        </Banner>
      ) : null}

      <Group justify="space-between" mt="xl" mb="xs">
        <Text fw={500}>Cases</Text>
        <Button variant="subtle" size="compact-sm" leftSection={<IconRefresh size={14} />} onClick={reload}>
          Refresh
        </Button>
      </Group>

      <ErrorBanner error={error} onRetry={reload} />

      {loading && !data ? (
        <LoadingBlock lines={6} />
      ) : (data?.cases.length ?? 0) === 0 ? (
        <Text c="dimmed" fz="sm">
          No cases yet. Examined files appear here.
        </Text>
      ) : (
        <Paper withBorder>
          <Group justify="space-between" p="sm" wrap="wrap">
            <div>
              <Text fw={500} fz="sm">
                {cases.length} case{cases.length === 1 ? '' : 's'}
              </Text>
              <Text fz="xs" c="dimmed">
                Select a case to open its examination record.
              </Text>
            </div>
            <TextInput
              aria-label="Search cases"
              placeholder="Filter by file name, case id or checksum"
              leftSection={<IconSearch size={14} />}
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              w={{ base: '100%', sm: 320 }}
            />
          </Group>
          <Table.ScrollContainer minWidth={820}>
            <Table highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  {COLUMNS.map((column) => (
                    <SortHeader
                      key={column.key}
                      column={column}
                      active={sort.key === column.key}
                      direction={sort.direction}
                      onClick={() => toggleSort(column.key)}
                    />
                  ))}
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {cases.map((item) => {
                  const band = presentBand(item.consensus.band)
                  return (
                    <Table.Tr
                      key={item.media_id}
                      className="vy-clickable"
                      onClick={() => navigate(`/case/${item.media_id}`)}
                    >
                      <Table.Td>
                        <Text fz="sm">{item.file_name}</Text>
                        <Text fz="xs" c="dimmed" ff="monospace">
                          {item.media_id} · {shortHash(item.checksum, 10, 6)}
                        </Text>
                      </Table.Td>
                      <Table.Td>{item.media_type}</Table.Td>
                      <Table.Td>
                        <Badge color={band.tagType} size="sm">
                          {band.label}
                        </Badge>
                      </Table.Td>
                      <Table.Td>
                        {item.findings_count
                          ? `${item.findings_count} ran · ${item.detectors_excluded} skipped`
                          : item.has_selection
                            ? `0 ran · ${item.detectors_excluded} skipped`
                            : 'not analysed'}
                      </Table.Td>
                      <Table.Td>{formatBytes(item.size_bytes)}</Table.Td>
                      <Table.Td>{formatRelative(item.ingested_at)}</Table.Td>
                      <Table.Td>
                        {item.reports.pdf ? 'PDF' : item.reports.html ? 'HTML' : '—'}
                      </Table.Td>
                    </Table.Tr>
                  )
                })}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          {cases.length === 0 && query ? (
            <Text c="dimmed" fz="sm" p="md">
              No case matches “{query}”.
            </Text>
          ) : null}
        </Paper>
      )}

    </div>
  )
}

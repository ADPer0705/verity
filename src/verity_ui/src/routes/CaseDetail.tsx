/**
 * One case: summary first, full examination record underneath.
 *
 * The action bar exposes every pipeline stage individually — the same
 * granularity the CLI gives (`extract`, `analyze`, `compile`, `report`,
 * `run`) — because re-running one stage against an existing workspace is the
 * normal way to work a case. While a job runs for this case every action is
 * disabled, and the page reattaches to that job after a reload. The active
 * tab (and, for Evidence, the selected detector) lives in the URL so a view
 * can be bookmarked or handed to a colleague.
 */

import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  ActionIcon,
  Anchor,
  Badge,
  Breadcrumbs,
  Button,
  Group,
  Menu,
  Modal,
  Notification,
  Tabs,
  Text,
  Title,
} from '@mantine/core'
import {
  IconChartBar,
  IconDotsVertical,
  IconFileText,
  IconFilter,
  IconPlayerPlay,
  IconRefresh,
  IconTrash,
} from '@tabler/icons-react'

import { ApiError, api } from '@/api/client'
import { useAsync, useJobStream } from '@/api/hooks'
import { JobConsole } from '@/components/JobConsole'
import { ErrorBanner, LoadingBlock } from '@/components/Primitives'
import { formatBytes } from '@/lib/format'
import { presentBand } from '@/lib/verdict'
import { RunPanel } from './RunPanel'
import type { RunConfig } from './RunPanel'
import { ActivityPanel } from './Activity'
import { AnalysisTab } from './tabs/AnalysisTab'
import { EvidenceTab } from './tabs/EvidenceTab'
import { FindingsTab } from './tabs/FindingsTab'
import { OverviewTab } from './tabs/OverviewTab'
import { ReportTab } from './tabs/ReportTab'
import { SelectionTab } from './tabs/SelectionTab'

const TABS = ['overview', 'analysis', 'selection', 'findings', 'evidence', 'report', 'activity']

export default function CaseDetail() {
  const { mediaId = '' } = useParams()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()

  const detail = useAsync(() => api.case(mediaId), [mediaId])
  const system = useAsync(() => api.system(), [])

  const [jobId, setJobId] = useState<string | null>(null)
  const [runOpen, setRunOpen] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const stream = useJobStream(jobId)
  const { reload } = detail

  const requestedTab = params.get('tab') ?? 'overview'
  const tab = TABS.includes(requestedTab) ? requestedTab : 'overview'
  const detectorParam = params.get('detector')

  const setTab = useCallback(
    (next: string | null, detector?: string | null) => {
      setParams(
        (current) => {
          const copy = new URLSearchParams(current)
          if (!next || next === 'overview') copy.delete('tab')
          else copy.set('tab', next)
          if (detector === null) copy.delete('detector')
          else if (detector) copy.set('detector', detector)
          return copy
        },
        { replace: true },
      )
    },
    [setParams],
  )

  // A finished job changes what is on disk; re-read the case once.
  useEffect(() => {
    if (stream.finished) reload()
  }, [stream.finished, reload])

  // Reattach to a job already running for this case (started earlier, from
  // another tab, or before a page reload) so progress is always visible and
  // nothing can start a second job against the same case meanwhile.
  useEffect(() => {
    let cancelled = false
    api
      .jobs({ mediaId, limit: 10 })
      .then((result) => {
        const running = result.jobs.find((job) => job.status === 'running' || job.status === 'pending')
        if (!cancelled && running) setJobId(running.id)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [mediaId])

  // Arriving from an upload (?run=1) opens the run dialog straight away.
  useEffect(() => {
    if (params.get('run') === '1') {
      setRunOpen(true)
      setParams(
        (current) => {
          const copy = new URLSearchParams(current)
          copy.delete('run')
          return copy
        },
        { replace: true },
      )
    }
  }, [params, setParams])

  const launch = useCallback(async (start: () => Promise<{ id: string }>) => {
    setBusy(true)
    setActionError(null)
    try {
      const job = await start()
      setJobId(job.id)
    } catch (cause) {
      setActionError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }, [])

  const runExamination = (config: RunConfig) => {
    setRunOpen(false)
    void launch(() =>
      api.run({
        media_id: mediaId,
        detectors: config.detectors,
        extractors: config.extractors,
        format: config.format,
        skip_report: config.skipReport,
        include_untrained: config.includeUntrained,
      }),
    )
  }

  const remove = async () => {
    try {
      await api.deleteCase(mediaId)
      navigate('/')
    } catch (cause) {
      setActionError(cause instanceof ApiError ? cause.message : String(cause))
      setDeleteOpen(false)
    }
  }

  if (detail.loading && !detail.data) {
    return (
      <div className="vy-page">
        <LoadingBlock lines={8} />
      </div>
    )
  }

  if (detail.error || !detail.data) {
    return (
      <div className="vy-page">
        <ErrorBanner error={detail.error ?? 'Case not found.'} onRetry={detail.reload} />
        <Button component={Link} to="/" variant="default">
          Back to cases
        </Button>
      </div>
    )
  }

  const data = detail.data
  const band = presentBand(data.consensus.band)
  const readiness = system.data?.readiness
  const canReportPdf = readiness?.can_report_pdf ?? true
  const pdfBlocker = readiness?.blockers.find((b) => b.capability === 'report-pdf')?.message
  const active = jobId !== null && !stream.finished
  const untrainedCount = data.findings.filter((finding) => !finding.trained).length

  return (
    <div className="vy-page">
      <Breadcrumbs mb="xs" fz="sm">
        <Anchor component={Link} to="/" fz="sm">
          Cases
        </Anchor>
        <Text fz="sm" c="dimmed">
          {data.media.file_name}
        </Text>
      </Breadcrumbs>

      <Group justify="space-between" align="flex-start" mb="md" wrap="wrap">
        <div>
          <Group gap="sm" wrap="wrap">
            <Title order={1} fz="1.375rem">
              {data.media.file_name}
            </Title>
            <Badge color={band.tagType} variant="filled">
              {band.label}
            </Badge>
          </Group>
          <Text fz="xs" c="dimmed" mt={4}>
            {data.media.media_type} · {formatBytes(data.media.size_bytes)} ·{' '}
            <span className="vy-mono">{data.media_id}</span>
          </Text>
        </div>

        <Group gap="xs" className="vy-no-print">
          <Button
            variant="default"
            size="sm"
            leftSection={<IconRefresh size={16} />}
            onClick={detail.reload}
          >
            Refresh
          </Button>
          <Button
            size="sm"
            leftSection={<IconPlayerPlay size={16} />}
            disabled={busy || active}
            onClick={() => setRunOpen(true)}
          >
            {active ? 'Examination running…' : 'Run examination'}
          </Button>
          <Menu position="bottom-end" withinPortal>
            <Menu.Target>
              <ActionIcon variant="default" size="lg" aria-label="More pipeline actions">
                <IconDotsVertical size={18} />
              </ActionIcon>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Label>Pipeline stages</Menu.Label>
              <Menu.Item
                leftSection={<IconFilter size={14} />}
                disabled={busy || active}
                onClick={() => void launch(() => api.extract(mediaId))}
              >
                Extract artifacts
              </Menu.Item>
              <Menu.Item
                leftSection={<IconChartBar size={14} />}
                disabled={busy || active}
                onClick={() => void launch(() => api.analyze(mediaId))}
              >
                Analyze and select detectors
              </Menu.Item>
              <Menu.Item
                leftSection={<IconFileText size={14} />}
                disabled={busy || active}
                onClick={() => void launch(() => api.compileFindings(mediaId))}
              >
                Compile findings
              </Menu.Item>
              <Menu.Item
                disabled={busy || active}
                onClick={() => void launch(() => api.report(mediaId, 'html'))}
              >
                Generate HTML report
              </Menu.Item>
              <Menu.Item
                disabled={busy || active || !canReportPdf}
                onClick={() => void launch(() => api.report(mediaId, 'pdf'))}
              >
                Generate PDF report
              </Menu.Item>
              <Menu.Divider />
              <Menu.Item
                color="red"
                leftSection={<IconTrash size={14} />}
                disabled={active}
                onClick={() => setDeleteOpen(true)}
              >
                Delete case
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </Group>
      </Group>

      {actionError ? (
        <Notification
          color="red"
          title="Could not start"
          withBorder
          mb="md"
          onClose={() => setActionError(null)}
        >
          {actionError}
        </Notification>
      ) : null}

      {jobId ? (
        <div style={{ marginBottom: '1rem' }}>
          <JobConsole
            job={stream.job}
            logs={stream.logs}
            onCancel={() => jobId && api.cancelJob(jobId).catch(() => undefined)}
          />
        </div>
      ) : null}

      <Tabs value={tab} onChange={(value) => setTab(value)} keepMounted={false}>
        <Tabs.List>
          <Tabs.Tab value="overview">Overview</Tabs.Tab>
          <Tabs.Tab value="analysis">Analysis</Tabs.Tab>
          <Tabs.Tab value="selection">Selection ({data.selection?.decisions.length ?? 0})</Tabs.Tab>
          <Tabs.Tab
            value="findings"
            rightSection={
              untrainedCount ? (
                <Badge color="red" variant="filled" size="xs" title="Untrained-head findings">
                  {untrainedCount}
                </Badge>
              ) : undefined
            }
          >
            Findings ({data.findings.length})
          </Tabs.Tab>
          <Tabs.Tab value="evidence">Evidence</Tabs.Tab>
          <Tabs.Tab value="report">Report</Tabs.Tab>
          <Tabs.Tab value="activity">Activity</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="overview" pt="lg">
          <OverviewTab detail={data} />
        </Tabs.Panel>
        <Tabs.Panel value="analysis" pt="lg">
          <AnalysisTab detail={data} />
        </Tabs.Panel>
        <Tabs.Panel value="selection" pt="lg">
          <SelectionTab detail={data} />
        </Tabs.Panel>
        <Tabs.Panel value="findings" pt="lg">
          <FindingsTab
            detail={data}
            onOpenEvidence={(detectorId) => setTab('evidence', detectorId)}
          />
        </Tabs.Panel>
        <Tabs.Panel value="evidence" pt="lg">
          <EvidenceTab
            detail={data}
            detectorId={detectorParam}
            onSelectDetector={(detectorId) => setTab('evidence', detectorId)}
            onOpenFinding={() => setTab('findings')}
          />
        </Tabs.Panel>
        <Tabs.Panel value="report" pt="lg">
          <ReportTab
            detail={data}
            canReportPdf={canReportPdf}
            pdfBlocker={pdfBlocker}
            busy={busy || active}
            onGenerate={(format) => void launch(() => api.report(mediaId, format))}
          />
        </Tabs.Panel>
        <Tabs.Panel value="activity" pt="lg">
          <ActivityPanel mediaId={mediaId} />
        </Tabs.Panel>
      </Tabs>

      <RunPanel
        open={runOpen}
        onClose={() => setRunOpen(false)}
        onSubmit={runExamination}
        mediaType={data.media.media_type}
        canReportPdf={canReportPdf}
      />

      <Modal
        opened={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        title={`Delete case ${data.media.file_name}?`}
        centered
      >
        <Text fz="sm">
          This removes the read-only working copy, its ingest record, every extracted artifact, all
          detector outputs and findings, and any generated report. It cannot be undone.
        </Text>
        <Text fz="sm" mt="sm">
          The original file at <span className="vy-mono vy-break">{data.media.original_path}</span>{' '}
          is not touched.
        </Text>
        <Group justify="flex-end" mt="lg">
          <Button variant="default" onClick={() => setDeleteOpen(false)}>
            Cancel
          </Button>
          <Button color="red" leftSection={<IconTrash size={16} />} onClick={remove}>
            Delete everything
          </Button>
        </Group>
      </Modal>
    </div>
  )
}

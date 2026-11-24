/**
 * Case overview — identity, integrity and the aggregate position.
 *
 * Integrity verification is a first-class action here rather than buried in a
 * menu: being able to re-derive the hash on demand, in front of someone, is
 * what makes the rest of the record defensible.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Anchor,
  Badge,
  Button,
  Grid,
  Group,
  List,
  Paper,
  Stack,
  Text,
  UnstyledButton,
} from '@mantine/core'
import {
  IconArchive,
  IconArrowRight,
  IconDownload,
  IconFingerprint,
  IconMicroscope,
  IconPhotoSearch,
} from '@tabler/icons-react'

import { ApiError, api, evidenceBundleUrl, mediaFileUrl } from '@/api/client'
import type { CaseDetail, IntegrityCheck } from '@/api/types'
import { ConsensusVerdict } from '@/components/ConsensusPanel'
import { unitLabels } from '@/components/FindingTables'
import { MediaPreview } from '@/components/MediaPreview'
import {
  Banner,
  CopyValue,
  KeyValue,
  MetricGrid,
  ProbabilityBar,
  SectionTitle,
} from '@/components/Primitives'
import { formatBytes, formatDuration, formatTimestamp, shortHash } from '@/lib/format'
import { CLASSIFICATION_TAG, standingLimitations } from '@/lib/verdict'

export function OverviewTab({
  detail,
  onOpenTab,
}: {
  detail: CaseDetail
  onOpenTab?: (tab: string, detectorId?: string) => void
}) {
  const { media, analysis, consensus, summary, forensics, findings } = detail
  const [check, setCheck] = useState<IntegrityCheck | null>(null)
  const [verifying, setVerifying] = useState(false)
  const [verifyError, setVerifyError] = useState<string | null>(null)

  const verify = async () => {
    setVerifying(true)
    setVerifyError(null)
    try {
      setCheck(await api.verify(detail.media_id))
    } catch (cause) {
      setVerifyError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setVerifying(false)
    }
  }

  const intact = check ? check.matches && check.size_matches : null
  const trainedFindings = [...findings]
    .filter((f) => f.trained)
    .sort((a, b) => b.mean_fake_probability - a.mean_fake_probability)
  const topForensicAlerts = (forensics?.results ?? [])
    .flatMap((r) =>
      r.observations
        .filter((o) => o.severity === 'warning' || o.severity === 'notice')
        .map((o) => ({ ...o, label: r.label })),
    )
    .sort((a, b) => (a.severity === 'warning' ? -1 : 1) - (b.severity === 'warning' ? -1 : 1))
    .slice(0, 5)

  return (
    <Grid gutter="xl">
      <Grid.Col span={{ base: 12, lg: 8 }}>
        <ConsensusVerdict consensus={consensus} />

        {consensus.total > 0 ? (
          <>
            <SectionTitle mt="1.5rem">What this does not tell you</SectionTitle>
            <List fz="sm" spacing={4} c="dimmed">
              {standingLimitations(consensus, media.media_type).map((line) => (
                <List.Item key={line}>{line}</List.Item>
              ))}
            </List>
            <Anchor component={Link} to={`/help/${detail.media_id}`} fz="sm" mt="xs" display="inline-block">
              How to preserve and report this file
            </Anchor>
          </>
        ) : null}

        <SectionTitle mt="1.5rem">Examination state</SectionTitle>
        <MetricGrid
          metrics={[
            {
              label: 'Detectors run',
              value: summary.findings_count,
              hint: `${summary.detectors_excluded} excluded by the selector`,
            },
            {
              label: 'Agreement',
              value: consensus.total ? `${Math.round(consensus.agreement * 100)}%` : '—',
              hint: consensus.total
                ? `${consensus.fake} manipulated · ${consensus.real} authentic · ${consensus.uncertain} undecided`
                : 'No findings compiled',
            },
            {
              label: 'Mean fake probability',
              value: consensus.mean_fake_probability?.toFixed(4) ?? '—',
              hint: 'Unweighted mean across evidence-grade detectors',
            },
            {
              label: 'Forensic signals',
              value: forensics
                ? `${forensics.counts?.warning ?? 0}W / ${forensics.counts?.notice ?? 0}N`
                : '—',
              hint: forensics
                ? `${forensics.results.length} model-free analyzers ran`
                : 'Forensics not yet run',
            },
          ]}
        />

        {trainedFindings.length > 0 ? (
          <>
            <Group justify="space-between" align="baseline" mt="1.5rem" mb="xs">
              <SectionTitle>Detector probability spectrum (evidence-grade)</SectionTitle>
              {onOpenTab ? (
                <Button
                  variant="subtle"
                  size="compact-xs"
                  rightSection={<IconArrowRight size={12} />}
                  onClick={() => onOpenTab('findings')}
                >
                  All findings ({findings.length})
                </Button>
              ) : null}
            </Group>
            <Paper withBorder p="sm" className="vy-card">
              <Stack gap={8}>
                {trainedFindings.map((finding) => {
                  const labels = unitLabels(finding.unit)
                  return (
                    <UnstyledButton
                      key={finding.detector_id}
                      className="vy-spectrum-row"
                      onClick={() => onOpenTab?.('evidence', finding.detector_id)}
                      title={`Inspect ${finding.detector_name} explainability evidence`}
                    >
                      <Group justify="space-between" wrap="nowrap" gap="sm">
                        <Group gap="xs" wrap="nowrap" style={{ minWidth: '11rem' }}>
                          <Badge
                            color={CLASSIFICATION_TAG[finding.classification]}
                            size="xs"
                            w={72}
                          >
                            {finding.classification}
                          </Badge>
                          <Text fz="sm" fw={500} truncate>
                            {finding.detector_name}
                          </Text>
                        </Group>
                        <div style={{ flex: 1 }}>
                          <ProbabilityBar value={finding.mean_fake_probability} />
                        </div>
                        <Text fz="xs" c="dimmed" ff="monospace" w="5.5rem" ta="right">
                          {finding.faces_analyzed ?? 0} {labels.plural}
                        </Text>
                      </Group>
                    </UnstyledButton>
                  )
                })}
              </Stack>
              <Text fz="xs" c="dimmed" mt="xs">
                Click any detector row to jump directly to its timelines, attention maps and
                per-unit scores in the Evidence Lab.
              </Text>
            </Paper>
          </>
        ) : null}

        {forensics ? (
          <>
            <Group justify="space-between" align="baseline" mt="1.5rem" mb="xs">
              <SectionTitle>Model-free forensic highlights</SectionTitle>
              {onOpenTab ? (
                <Button
                  variant="subtle"
                  size="compact-xs"
                  leftSection={<IconMicroscope size={12} />}
                  rightSection={<IconArrowRight size={12} />}
                  onClick={() => onOpenTab('forensics')}
                >
                  Open Forensics tab
                </Button>
              ) : null}
            </Group>
            <Paper withBorder p="sm" className="vy-card">
              {topForensicAlerts.length > 0 ? (
                <Stack gap={6}>
                  {topForensicAlerts.map((obs, idx) => (
                    <Group key={idx} gap="xs" align="flex-start" wrap="nowrap">
                      <Badge
                        color={obs.severity === 'warning' ? 'red' : 'orange'}
                        variant={obs.severity === 'warning' ? 'filled' : 'light'}
                        size="xs"
                        mt={2}
                        style={{ flexShrink: 0 }}
                      >
                        {obs.severity}
                      </Badge>
                      <Text fz="sm" lh={1.45}>
                        <b>{obs.label}:</b> {obs.statement}
                      </Text>
                    </Group>
                  ))}
                </Stack>
              ) : (
                <Text fz="sm" c="dimmed">
                  No warnings or notices raised across {forensics.results.length} model-free
                  analyzers ({forensics.counts?.info ?? 0} baseline measurements recorded).
                </Text>
              )}
            </Paper>
          </>
        ) : null}

        <SectionTitle mt="1.5rem">Media identity</SectionTitle>
        <KeyValue
          rows={[
            ['File name', media.file_name],
            ['Case id', <CopyValue key="id" value={detail.media_id} label="case id" />],
            ['Media type', `${media.media_type}${media.mime_type ? ` · ${media.mime_type}` : ''}`],
            ['Size', `${formatBytes(media.size_bytes)} (${media.size_bytes.toLocaleString()} bytes)`],
            [
              'SHA-256',
              <CopyValue
                key="sum"
                value={media.checksum}
                label="checksum"
                truncate={shortHash(media.checksum, 24, 12)}
              />,
            ],
            [
              'Source',
              /\/verity_upload_[^/]+\//.test(media.original_path) ? (
                'Uploaded through the web console'
              ) : (
                <span key="src" className="vy-mono vy-break">{media.original_path}</span>
              ),
            ],
            ['Working copy', <span key="copy" className="vy-mono vy-break">{media.working_copy}</span>],
            ['Ingested', formatTimestamp(summary.ingested_at)],
            ...(analysis?.media.duration_seconds != null
              ? ([['Duration', formatDuration(analysis.media.duration_seconds)]] as [string, string][])
              : []),
            ...(analysis?.media.width && analysis?.media.height
              ? ([['Resolution', `${analysis.media.width} × ${analysis.media.height}`]] as [
                  string,
                  string,
                ][])
              : []),
          ]}
        />

        <SectionTitle mt="1.5rem">Integrity</SectionTitle>
        <Text fz="sm" c="dimmed" mb="sm" maw="42rem">
          Recomputes the SHA-256 of the working copy the detectors read and compares it against the
          hash recorded at ingest. Run this before citing the findings.
        </Text>
        <Group gap="xs" wrap="wrap">
          <Button
            variant="default"
            leftSection={<IconFingerprint size={16} />}
            onClick={verify}
            disabled={verifying}
          >
            {verifying ? 'Verifying…' : 'Verify integrity now'}
          </Button>
          <Button
            variant="light"
            leftSection={<IconArchive size={16} />}
            component="a"
            href={evidenceBundleUrl(detail.media_id)}
          >
            Export evidence bundle (.zip)
          </Button>
          <Button
            variant="subtle"
            color="gray"
            leftSection={<IconDownload size={16} />}
            component="a"
            href={mediaFileUrl(detail.media_id, true)}
          >
            Download working copy
          </Button>
        </Group>

        {verifyError ? (
          <Banner color="red" title="Verification failed" mt="sm">
            {verifyError}
          </Banner>
        ) : null}

        {check ? (
          <div style={{ marginTop: '1rem' }}>
            <Banner
              color={intact ? 'teal' : 'red'}
              title={
                intact ? 'Working copy is unaltered' : 'Working copy does not match the ingest record'
              }
              mb="sm"
            >
              {intact
                ? `Recomputed hash matches the value recorded at ingest. Verified ${formatTimestamp(check.verified_at)}.`
                : 'The bytes on disk differ from what was admitted. Do not rely on findings derived from this copy.'}
            </Banner>
            <KeyValue
              rows={[
                ['Expected', <span key="e" className="vy-mono vy-break">{check.expected_checksum}</span>],
                ['Recomputed', <span key="a" className="vy-mono vy-break">{check.actual_checksum}</span>],
                [
                  'Size',
                  check.size_matches
                    ? `${check.working_copy_size.toLocaleString()} bytes — matches`
                    : `${check.working_copy_size.toLocaleString()} bytes on disk vs ${check.recorded_size.toLocaleString()} recorded`,
                ],
                [
                  'Original at source path',
                  check.original_state === 'absent' ? (
                    <Text key="o" component="span" c="dimmed">
                      Not present — the source file has moved or been removed since ingest.
                    </Text>
                  ) : check.original_matches ? (
                    <Badge key="o" color="teal" size="sm">
                      Still matches
                    </Badge>
                  ) : (
                    <Badge key="o" color="red" size="sm">
                      Differs from ingest
                    </Badge>
                  ),
                ],
              ]}
            />
          </div>
        ) : null}
      </Grid.Col>

      <Grid.Col span={{ base: 12, lg: 4 }}>
        <SectionTitle>Media</SectionTitle>
        <MediaPreview mediaId={detail.media_id} mediaType={media.media_type} fileName={media.file_name} />
        <Text fz="xs" c="dimmed" mt="xs">
          Played from the read-only working copy in the workspace, not the source path.
        </Text>
        {onOpenTab && findings.length > 0 ? (
          <Button
            variant="light"
            fullWidth
            mt="md"
            leftSection={<IconPhotoSearch size={16} />}
            onClick={() => onOpenTab('evidence')}
          >
            Open Evidence Lab ({findings.length} detector{findings.length === 1 ? '' : 's'})
          </Button>
        ) : null}
      </Grid.Col>
    </Grid>
  )
}

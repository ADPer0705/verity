/**
 * Case overview — identity, integrity and the aggregate position.
 *
 * Integrity verification is a first-class action here rather than buried in a
 * menu: being able to re-derive the hash on demand, in front of someone, is
 * what makes the rest of the record defensible.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Anchor, Badge, Button, Grid, Group, List, Text } from '@mantine/core'
import { IconDownload, IconFingerprint } from '@tabler/icons-react'

import { ApiError, api, mediaFileUrl } from '@/api/client'
import type { CaseDetail, IntegrityCheck } from '@/api/types'
import { ConsensusVerdict } from '@/components/ConsensusPanel'
import { MediaPreview } from '@/components/MediaPreview'
import { Banner, CopyValue, KeyValue, MetricGrid, SectionTitle } from '@/components/Primitives'
import { formatBytes, formatDuration, formatTimestamp, shortHash } from '@/lib/format'
import { standingLimitations } from '@/lib/verdict'

export function OverviewTab({ detail }: { detail: CaseDetail }) {
  const { media, analysis, consensus, summary } = detail
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
              label: 'Faces detected',
              value: analysis?.media.face_count ?? '—',
              hint: analysis
                ? `across ${analysis.media.sampled_frames} sampled frames`
                : 'Media not yet characterized',
            },
          ]}
        />

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
        <Group gap="xs">
          <Button
            variant="default"
            leftSection={<IconFingerprint size={16} />}
            onClick={verify}
            disabled={verifying}
          >
            {verifying ? 'Verifying…' : 'Verify integrity now'}
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
      </Grid.Col>
    </Grid>
  )
}

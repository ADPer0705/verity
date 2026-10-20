/**
 * Evidence — what each detector produced, and what it was fed.
 *
 * Two layers. "Detector outputs" is the explainability record per method:
 * timeline, scored face crops, attention maps, reconstructions, per-face
 * scores, raw JSON and logs. "Extracted artifacts" is the intermediate
 * material the detectors consumed: being able to look at the actual face
 * crops and sampled frames is how an examiner distinguishes "the model was
 * wrong" from "the model was shown the wrong thing".
 */

import { useMemo } from 'react'
import { Accordion, Anchor, Badge, Grid, Group, NavLink, Paper, Select, Text } from '@mantine/core'
import { IconLayersSubtract } from '@tabler/icons-react'

import { api, fileUrl } from '@/api/client'
import { useAsync } from '@/api/hooks'
import type { ArtifactGroup, CaseDetail, Finding } from '@/api/types'
import { DetectorEvidence } from '@/components/DetectorEvidence'
import { ImageGallery } from '@/components/ImageGallery'
import { EmptyState, SectionTitle } from '@/components/Primitives'
import { formatBytes, titleCase } from '@/lib/format'
import { CLASSIFICATION_TAG } from '@/lib/verdict'

const IMAGE_SUFFIXES = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'])

function GroupPanel({ mediaId, group }: { mediaId: string; group: ArtifactGroup }) {
  const images = group.files.filter((file) => IMAGE_SUFFIXES.has(file.suffix))
  const others = group.files.filter((file) => !IMAGE_SUFFIXES.has(file.suffix))

  return (
    <div>
      {images.length ? (
        <ImageGallery
          mediaId={mediaId}
          images={images.map((file) => ({ ...file, caption: file.name.replace(/\.[^.]+$/, '') }))}
          layout={group.group.includes('face') ? 'faces' : 'maps'}
        />
      ) : null}

      {others.length ? (
        <ul style={{ listStyle: 'none', padding: 0, margin: images.length ? '1rem 0 0' : 0 }}>
          {others.map((file) => (
            <li
              key={file.path}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: '1rem',
                padding: '0.4rem 0',
                borderBottom: '1px solid var(--mantine-color-default-border)',
              }}
            >
              <Anchor
                href={fileUrl(mediaId, file.path)}
                target="_blank"
                rel="noreferrer"
                ff="monospace"
                fz="xs"
                className="vy-break"
              >
                {file.name}
              </Anchor>
              <Text fz="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                {formatBytes(file.size_bytes)}
              </Text>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function orderFindings(findings: Finding[]): Finding[] {
  return [...findings].sort(
    (a, b) => Number(b.trained) - Number(a.trained) || a.detector_name.localeCompare(b.detector_name),
  )
}

export function EvidenceTab({
  detail,
  detectorId,
  onSelectDetector,
  onOpenFinding,
}: {
  detail: CaseDetail
  detectorId: string | null
  onSelectDetector: (detectorId: string) => void
  onOpenFinding: () => void
}) {
  const coverage = useAsync(() => api.coverage().catch(() => null), [])
  const findings = useMemo(() => orderFindings(detail.findings), [detail.findings])
  const selected =
    findings.find((finding) => finding.detector_id === detectorId) ?? findings[0] ?? null

  const total = detail.artifacts.reduce((sum, group) => sum + group.count, 0)

  return (
    <div>
      {/* ---------------- detector outputs ---------------- */}
      <SectionTitle>Detector outputs</SectionTitle>
      {findings.length === 0 ? (
        <Text fz="sm" c="dimmed" mb="xl">
          No detector has produced a finding for this case yet, so there is no explainability
          record to browse.
        </Text>
      ) : (
        <Grid gutter="md" mb="xl">
          <Grid.Col span={{ base: 12, md: 3 }}>
            <Select
              hiddenFrom="md"
              aria-label="Detector"
              data={findings.map((finding) => ({
                value: finding.detector_id,
                label: `${finding.detector_name} — ${finding.classification}${finding.trained ? '' : ' (untrained)'}`,
              }))}
              value={selected?.detector_id ?? null}
              onChange={(value) => value && onSelectDetector(value)}
            />
            <Paper withBorder visibleFrom="md" p={4}>
              {findings.map((finding) => (
                <NavLink
                  key={finding.detector_id}
                  active={selected?.detector_id === finding.detector_id}
                  onClick={() => onSelectDetector(finding.detector_id)}
                  label={
                    <Text fz="sm" truncate>
                      {finding.detector_name}
                    </Text>
                  }
                  description={
                    <Group gap={4} mt={2}>
                      <Badge color={CLASSIFICATION_TAG[finding.classification]} size="xs">
                        {finding.classification}
                      </Badge>
                      {!finding.trained ? (
                        <Badge color="red" variant="filled" size="xs">
                          {finding.tier}
                        </Badge>
                      ) : null}
                      <Text fz="xs" c="dimmed" ff="monospace">
                        {finding.mean_fake_probability.toFixed(3)}
                      </Text>
                    </Group>
                  }
                />
              ))}
            </Paper>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 9 }}>
            {selected ? (
              <Paper withBorder p="md">
                <DetectorEvidence
                  key={selected.detector_id}
                  mediaId={detail.media_id}
                  finding={selected}
                  families={coverage.data?.families}
                  onOpenFinding={onOpenFinding}
                />
              </Paper>
            ) : null}
          </Grid.Col>
        </Grid>
      )}

      {/* ---------------- extracted artifacts ---------------- */}
      <Group justify="space-between" align="flex-end" mb="sm">
        <div>
          <SectionTitle>Extracted artifacts</SectionTitle>
          <Text fz="sm" c="dimmed">
            {detail.artifacts.length
              ? `${total.toLocaleString()} file${total === 1 ? '' : 's'} across ${detail.artifacts.length} group${detail.artifacts.length === 1 ? '' : 's'} — the material the detectors were fed.`
              : 'Nothing extracted yet.'}
          </Text>
        </div>
      </Group>

      {!detail.artifacts.length ? (
        <EmptyState
          icon={<IconLayersSubtract size={26} />}
          title="No artifacts extracted yet"
          body="Run Extract (or a full examination) to produce sampled frames, face crops, container metadata and the audio track under workspace/artifacts/."
        />
      ) : (
        <Accordion variant="contained" multiple>
          {detail.artifacts.map((group) => (
            <Accordion.Item key={group.group} value={group.group}>
              <Accordion.Control>
                <Group gap="sm">
                  <Text fw={500} fz="sm">
                    {group.group === '.' ? 'Root' : titleCase(group.group)}
                  </Text>
                  <Badge variant="outline" color="gray" size="sm">
                    {group.count}
                  </Badge>
                </Group>
              </Accordion.Control>
              <Accordion.Panel>
                <GroupPanel mediaId={detail.media_id} group={group} />
              </Accordion.Panel>
            </Accordion.Item>
          ))}
        </Accordion>
      )}
    </div>
  )
}

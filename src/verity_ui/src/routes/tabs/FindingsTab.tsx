/**
 * Per-detector findings with full drill-down, grouped by what they examined.
 *
 * The coverage strip answers the question a reviewer asks first — which kinds
 * of manipulation were actually tested for on this media, and which were
 * not — before the per-detector table says what each method concluded. Each
 * finding keeps its own provenance; the aggregate is for orientation. The
 * detail panels fetch lazily, when opened.
 */

import { useMemo, useState } from 'react'
import {
  Accordion,
  ActionIcon,
  Badge,
  Blockquote,
  Group,
  List,
  Paper,
  Stack,
  Table,
  Tabs,
  Text,
  Tooltip,
} from '@mantine/core'
import { IconPhotoSearch, IconSearch } from '@tabler/icons-react'

import { api } from '@/api/client'
import { useAsync } from '@/api/hooks'
import type { CaseDetail, CoverageFamily, Finding } from '@/api/types'
import { ExplainabilityPanel, LogsPanel, RawPanel } from '@/components/DetectorEvidence'
import { PerFaceTable, RegionEnergyTable } from '@/components/FindingTables'
import {
  EmptyState,
  FamilyChips,
  KeyValue,
  ProbabilityBar,
  SectionTitle,
  TierBadge,
} from '@/components/Primitives'
import { formatPercent, formatTimestamp } from '@/lib/format'
import { FAMILIES, LEAN_COLOR, LEAN_LABEL, caseCoverage } from '@/lib/taxonomy'
import { CLASSIFICATION_TAG } from '@/lib/verdict'

function CoverageStrip({ detail, families }: { detail: CaseDetail; families: CoverageFamily[] }) {
  const rows = useMemo(
    () => caseCoverage(detail.findings, detail.media.media_type, families),
    [detail.findings, detail.media.media_type, families],
  )
  const examined = rows.filter((row) => row.examined)

  return (
    <div>
      <SectionTitle>Coverage for this {detail.media.media_type === 'unknown' ? 'file' : detail.media.media_type}</SectionTitle>
      <Text fz="sm" c="dimmed" mb="sm" maw="60rem">
        {examined.length} of {rows.length} manipulation famil{rows.length === 1 ? 'y' : 'ies'}{' '}
        relevant to this media type {examined.length === 1 ? 'was' : 'were'} examined by at least
        one evidence-grade detector. A family's lean is the majority classification among the
        evidence-grade detectors that target it; untrained heads are listed but never counted.
      </Text>
      <div className="vy-coverage">
        {rows.map((row) => {
          const state = !row.examined
            ? 'none'
            : row.lean === 'manipulated'
              ? 'fake'
              : row.lean === 'authentic'
                ? 'real'
                : 'uncertain'
          return (
            <div key={row.family.id} className={`vy-family vy-family--${state}`}>
              <Group justify="space-between" align="flex-start" wrap="nowrap" gap="xs">
                <Text fw={500} fz="sm">
                  {row.family.label}
                </Text>
                {row.examined && row.lean ? (
                  <Badge color={LEAN_COLOR[row.lean]} size="sm">
                    {LEAN_LABEL[row.lean]}
                  </Badge>
                ) : (
                  <Badge color="gray" variant="outline" size="sm">
                    Not examined
                  </Badge>
                )}
              </Group>
              {row.examined ? (
                <>
                  <Text fz="xs" c="dimmed" mt={4}>
                    {row.trained.length} evidence-grade detector{row.trained.length === 1 ? '' : 's'}{' '}
                    ran · {row.fake} manipulated · {row.real} authentic · {row.uncertain} undecided
                  </Text>
                  <Text fz="xs" ff="monospace" mt={4} className="vy-break">
                    {row.trained.map((finding) => finding.detector_id).join(', ')}
                  </Text>
                </>
              ) : (
                <Text fz="xs" c="dimmed" mt={4}>
                  No evidence-grade detector targeting this family produced a finding.
                </Text>
              )}
              {row.untrained.length ? (
                <Text fz="xs" c="red" mt={4}>
                  {row.untrained.length} untrained-head finding{row.untrained.length === 1 ? '' : 's'}{' '}
                  not counted.
                </Text>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function FindingPanel({
  mediaId,
  finding,
  families,
}: {
  mediaId: string
  finding: Finding
  families: CoverageFamily[]
}) {
  return (
    <Tabs defaultValue="reasoning" keepMounted={false} variant="outline">
      <Tabs.List>
        <Tabs.Tab value="reasoning">Reasoning</Tabs.Tab>
        <Tabs.Tab value="faces">Per-face ({finding.per_face.length})</Tabs.Tab>
        <Tabs.Tab value="explainability">Explainability</Tabs.Tab>
        <Tabs.Tab value="raw">Raw output</Tabs.Tab>
        <Tabs.Tab value="logs">Logs</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="reasoning" pt="md">
        <Blockquote p="md" mb="md" color={CLASSIFICATION_TAG[finding.classification]}>
          <Text fz="sm" lh={1.6}>
            {finding.reasoning.summary}
          </Text>
        </Blockquote>

        {finding.reasoning.confidence_factors.length ? (
          <>
            <SectionTitle>Confidence factors</SectionTitle>
            <List fz="sm" spacing={4} mb="md">
              {finding.reasoning.confidence_factors.map((factor) => (
                <List.Item key={factor}>{factor}</List.Item>
              ))}
            </List>
          </>
        ) : null}

        {finding.reasoning.evidence?.length ? (
          <>
            <SectionTitle>Evidence cited</SectionTitle>
            <List fz="sm" spacing={4} mb="md">
              {finding.reasoning.evidence.map((item) => (
                <List.Item key={item}>{item}</List.Item>
              ))}
            </List>
          </>
        ) : null}

        <SectionTitle>Key figures</SectionTitle>
        <KeyValue
          rows={[
            ['Classification', finding.classification],
            ['Classification head', <TierBadge key="tier" tier={finding.tier} />],
            ['Targets', <FamilyChips key="t" targets={finding.targets} families={families} size="sm" />],
            ['Mean fake probability', finding.mean_fake_probability.toFixed(4)],
            ['Detector confidence', finding.confidence.toFixed(4)],
            ['Faces analyzed', finding.faces_analyzed?.toLocaleString() ?? '—'],
            ...(finding.stats.min != null
              ? ([['Score range', `[${finding.stats.min}, ${finding.stats.max}]`]] as [string, string][])
              : []),
            ...(finding.stats.above_threshold != null && finding.stats.count
              ? ([
                  ['Above 0.7 threshold', `${finding.stats.above_threshold} / ${finding.stats.count}`],
                ] as [string, string][])
              : []),
            ...(finding.sampling.coverage_pct != null
              ? ([['Video sampling coverage', `${finding.sampling.coverage_pct}%`]] as [string, string][])
              : []),
            ['Detector version', finding.detector_version],
            ['Produced', formatTimestamp(finding.timestamp)],
          ]}
        />

        <div style={{ marginTop: '1rem' }}>
          <RegionEnergyTable finding={finding} />
        </div>
      </Tabs.Panel>

      <Tabs.Panel value="faces" pt="md">
        <PerFaceTable rows={finding.per_face} />
      </Tabs.Panel>

      <Tabs.Panel value="explainability" pt="md">
        <ExplainabilityPanel mediaId={mediaId} finding={finding} />
      </Tabs.Panel>

      <Tabs.Panel value="raw" pt="md">
        <RawPanel mediaId={mediaId} finding={finding} />
      </Tabs.Panel>

      <Tabs.Panel value="logs" pt="md">
        <LogsPanel mediaId={mediaId} finding={finding} />
      </Tabs.Panel>
    </Tabs>
  )
}

export function FindingsTab({
  detail,
  onOpenEvidence,
}: {
  detail: CaseDetail
  onOpenEvidence: (detectorId: string) => void
}) {
  const coverage = useAsync(() => api.coverage().catch(() => null), [])
  const families = coverage.data?.families ?? FAMILIES
  const [opened, setOpened] = useState<string[]>([])

  // Evidence-grade first; untrained heads sorted after, then by name.
  const findings = useMemo(
    () =>
      [...detail.findings].sort(
        (a, b) =>
          Number(b.trained) - Number(a.trained) || a.detector_name.localeCompare(b.detector_name),
      ),
    [detail.findings],
  )

  if (!findings.length) {
    return (
      <EmptyState
        icon={<IconSearch size={26} />}
        title="No findings compiled for this case"
        body="Run a full examination, or use Compile if detector outputs already exist in the workspace. Findings are read from workspace/output/<case>_<detector>/finding.json."
      />
    )
  }

  const untrained = findings.filter((finding) => !finding.trained).length

  return (
    <Stack gap="xl">
      <CoverageStrip detail={detail} families={families} />

      <div>
        <SectionTitle>Findings by detector</SectionTitle>
        <Text fz="sm" c="dimmed" mb="sm">
          One row per detector that produced a finding. Figures match the exported report exactly.
          {untrained
            ? ` ${untrained} finding${untrained === 1 ? '' : 's'} came from untrained heads (red) and sit${untrained === 1 ? 's' : ''} outside the consensus.`
            : ''}
        </Text>
        <Paper withBorder>
          <Table.ScrollContainer minWidth={900}>
            <Table highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Method</Table.Th>
                  <Table.Th>Targets</Table.Th>
                  <Table.Th>Classification</Table.Th>
                  <Table.Th>Faces</Table.Th>
                  <Table.Th w="15rem">Mean fake probability</Table.Th>
                  <Table.Th>Confidence</Table.Th>
                  <Table.Th aria-label="Evidence" w={48} />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {findings.map((finding) => (
                  <Table.Tr key={finding.detector_id}>
                    <Table.Td>
                      <Text fz="sm">{finding.detector_name}</Text>
                      <Text fz="xs" c="dimmed" ff="monospace">
                        {finding.detector_id} v{finding.detector_version}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <FamilyChips targets={finding.targets} families={families} />
                    </Table.Td>
                    <Table.Td>
                      <Group gap={4} wrap="nowrap">
                        <Badge color={CLASSIFICATION_TAG[finding.classification]} size="sm">
                          {finding.classification}
                        </Badge>
                        {!finding.trained ? <TierBadge tier={finding.tier} /> : null}
                      </Group>
                    </Table.Td>
                    <Table.Td>{finding.faces_analyzed?.toLocaleString() ?? '—'}</Table.Td>
                    <Table.Td>
                      <ProbabilityBar value={finding.mean_fake_probability} />
                    </Table.Td>
                    <Table.Td ff="monospace">{finding.confidence.toFixed(4)}</Table.Td>
                    <Table.Td>
                      <Tooltip label="Browse this detector's evidence">
                        <ActionIcon
                          variant="subtle"
                          color="gray"
                          size="sm"
                          aria-label={`Evidence for ${finding.detector_name}`}
                          onClick={() => onOpenEvidence(finding.detector_id)}
                        >
                          <IconPhotoSearch size={16} />
                        </ActionIcon>
                      </Tooltip>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </Paper>

        {detail.consensus.disagreement ? (
          <Text fz="xs" c="dimmed" mt="xs" maw="52rem">
            These detectors did not agree. Before citing any one of them, check whether the
            disagreement tracks a property of the media — heavy compression, small faces, few
            sampled frames — in the Analysis tab.
          </Text>
        ) : null}
      </div>

      <div>
        <SectionTitle>Detail by method</SectionTitle>
        <Accordion variant="contained" multiple value={opened} onChange={setOpened}>
          {findings.map((finding) => (
            <Accordion.Item key={finding.detector_id} value={finding.detector_id}>
              <Accordion.Control>
                <Group gap="sm" wrap="wrap">
                  <Text fw={500} fz="sm">
                    {finding.detector_name}
                  </Text>
                  <Badge color={CLASSIFICATION_TAG[finding.classification]} size="sm">
                    {finding.classification}
                  </Badge>
                  {!finding.trained ? <TierBadge tier={finding.tier} /> : null}
                  <Text fz="xs" c="dimmed">
                    mean {finding.mean_fake_probability.toFixed(4)} · confidence{' '}
                    {formatPercent(finding.confidence, 1)}
                  </Text>
                </Group>
              </Accordion.Control>
              <Accordion.Panel>
                {opened.includes(finding.detector_id) ? (
                  <FindingPanel mediaId={detail.media_id} finding={finding} families={families} />
                ) : null}
              </Accordion.Panel>
            </Accordion.Item>
          ))}
        </Accordion>
      </div>
    </Stack>
  )
}

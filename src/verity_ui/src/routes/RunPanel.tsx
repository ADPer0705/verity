/**
 * Run dialog — decides which detectors examine this file.
 *
 * Two modes. "Recommended" lets the selector pick every evidence-grade
 * detector that can read the file and records why each one ran or was
 * skipped. "Choose detectors" is for picking exactly what runs, e.g. to fit
 * the hardware at hand or to re-run one method; the list is pre-filled with
 * the recommended set and limited to detectors that can read this media
 * type. Every control has a fixed label: the label states what the control
 * does, its state is shown by the control itself.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  Accordion,
  Badge,
  Button,
  Checkbox,
  Group,
  Modal,
  ScrollArea,
  SegmentedControl,
  Stack,
  Table,
  Text,
} from '@mantine/core'

import { api } from '@/api/client'
import { useAsync } from '@/api/hooks'
import type { DetectorEntry } from '@/api/types'
import { Banner, FamilyChips, TierBadge } from '@/components/Primitives'
import { FAMILIES, familyLabel } from '@/lib/taxonomy'

const EXTRACTORS: { key: string; label: string }[] = [
  { key: 'frame', label: 'Frames (sampled video frames)' },
  { key: 'face', label: 'Faces (needed by face-based detectors)' },
  { key: 'metadata', label: 'Metadata (container and stream information)' },
  { key: 'audio', label: 'Audio (soundtrack, needed by voice detectors)' },
]

export interface RunConfig {
  detectors: string[] | null
  extractors: string[] | null
  format: 'pdf' | 'html'
  skipReport: boolean
  includeUntrained: boolean
}

type Mode = 'recommended' | 'choose'
type ReportChoice = 'pdf' | 'html' | 'none'

export function RunPanel({
  open,
  onClose,
  onSubmit,
  mediaType,
  canReportPdf = true,
}: {
  open: boolean
  onClose: () => void
  onSubmit: (config: RunConfig) => void
  mediaType: string
  canReportPdf?: boolean
}) {
  const catalog = useAsync(() => api.detectors(), [])
  const coverage = useAsync(() => api.coverage(), [])
  const families = coverage.data?.families ?? FAMILIES

  const [mode, setMode] = useState<Mode>('recommended')
  const [includeUntrained, setIncludeUntrained] = useState(false)
  const [chosen, setChosen] = useState<string[]>([])
  const [report, setReport] = useState<ReportChoice>(canReportPdf ? 'pdf' : 'html')
  const [extractors, setExtractors] = useState<string[]>(EXTRACTORS.map((e) => e.key))

  const all: DetectorEntry[] = catalog.data?.detectors ?? []
  /** Detectors that can read this media type, evidence-grade first. */
  const compatible = useMemo(
    () =>
      all
        .filter((d) => d.media_types.includes(mediaType))
        .sort(
          (a, b) =>
            Number(b.trained) - Number(a.trained) ||
            Number(b.weights.available) - Number(a.weights.available) ||
            a.name.localeCompare(b.name),
        ),
    [all, mediaType],
  )
  const recommended = useMemo(
    () => compatible.filter((d) => d.trained && d.weights.available).map((d) => d.detector_id),
    [compatible],
  )
  const hiddenCount = all.length - compatible.length

  // Every time the dialog opens it starts from the same, predictable state.
  useEffect(() => {
    if (!open) return
    setMode('recommended')
    setIncludeUntrained(false)
    setChosen(recommended)
    setReport(canReportPdf ? 'pdf' : 'html')
    setExtractors(EXTRACTORS.map((e) => e.key))
  }, [open, recommended, canReportPdf])

  const familyGroups = useMemo(() => {
    const ids = new Set(compatible.flatMap((d) => d.targets ?? []))
    return families
      .filter((family) => ids.has(family.id))
      .map((family) => ({
        id: family.id,
        label: familyLabel(family.id, families),
        ids: compatible
          .filter((d) => d.trained && d.weights.available && (d.targets ?? []).includes(family.id))
          .map((d) => d.detector_id),
      }))
      .filter((group) => group.ids.length > 0)
  }, [compatible, families])

  const setMany = (ids: string[], on: boolean) =>
    setChosen((list) =>
      on ? Array.from(new Set([...list, ...ids])) : list.filter((id) => !ids.includes(id)),
    )

  const submit = () =>
    onSubmit({
      detectors: mode === 'choose' ? chosen : null,
      extractors: extractors.length === EXTRACTORS.length ? null : extractors,
      format: report === 'none' ? 'html' : report,
      skipReport: report === 'none',
      includeUntrained: mode === 'recommended' && includeUntrained,
    })

  const chosenUntrained = all.filter((d) => chosen.includes(d.detector_id) && !d.trained).length
  const chosenMissing = all.filter(
    (d) => chosen.includes(d.detector_id) && !d.weights.available,
  ).length

  return (
    <Modal opened={open} onClose={onClose} title="Run examination" size="xl">
      <Stack gap="lg">
        <div>
          <Text fw={500} fz="sm" mb={6}>
            Detectors
          </Text>
          <SegmentedControl
            fullWidth
            value={mode}
            onChange={(value) => setMode(value as Mode)}
            data={[
              { value: 'recommended', label: `Recommended (${recommended.length})` },
              { value: 'choose', label: 'Choose detectors' },
            ]}
          />
        </div>

        {mode === 'recommended' ? (
          <Stack gap="sm">
            <Text fz="sm" c="dimmed">
              Verity runs every evidence-grade detector that can read this {mediaType}: right
              media type, faces present where needed, long enough for clip-based methods, and
              weights on disk. The case's Selection tab records why each detector ran or was
              skipped. Detectors run one after another.
            </Text>
            <Checkbox
              checked={includeUntrained}
              onChange={(event) => setIncludeUntrained(event.currentTarget.checked)}
              label="Also run research-only detectors"
              description="Detectors without a deepfake-trained head produce noise. They are labelled in the results and never counted toward the summary."
            />
          </Stack>
        ) : (
          <Stack gap="xs">
            <Group gap="xs">
              <Button variant="default" size="compact-sm" onClick={() => setChosen(recommended)}>
                Recommended set
              </Button>
              <Button variant="default" size="compact-sm" onClick={() => setChosen([])}>
                Clear
              </Button>
              <Text fz="xs" c="dimmed">
                Add every evidence-grade detector for:
              </Text>
              {familyGroups.map((group) => (
                <Button
                  key={group.id}
                  variant="light"
                  size="compact-xs"
                  onClick={() => setMany(group.ids, true)}
                >
                  {group.label} ({group.ids.length})
                </Button>
              ))}
            </Group>

            <ScrollArea.Autosize mah="22rem" type="auto" offsetScrollbars>
              <Table verticalSpacing={4} highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th w={36} />
                    <Table.Th>Detector</Table.Th>
                    <Table.Th>Looks for</Table.Th>
                    <Table.Th>Status</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {compatible.map((detector) => {
                    const checked = chosen.includes(detector.detector_id)
                    return (
                      <Table.Tr
                        key={detector.detector_id}
                        className="vy-clickable"
                        onClick={() => setMany([detector.detector_id], !checked)}
                      >
                        <Table.Td>
                          <Checkbox
                            checked={checked}
                            onChange={() => undefined}
                            aria-label={`Run ${detector.name}`}
                            tabIndex={-1}
                          />
                        </Table.Td>
                        <Table.Td>
                          <Text fz="sm">{detector.name}</Text>
                          <Text fz="xs" c="dimmed" ff="monospace">
                            {detector.detector_id}
                          </Text>
                        </Table.Td>
                        <Table.Td>
                          <FamilyChips targets={detector.targets} families={families} />
                        </Table.Td>
                        <Table.Td>
                          <Group gap={4}>
                            {detector.trained ? null : <TierBadge tier={detector.tier} size="xs" />}
                            {detector.weights.available ? null : (
                              <Badge color="orange" size="xs">
                                weights missing
                              </Badge>
                            )}
                            {detector.trained && detector.weights.available ? (
                              <Badge color="teal" size="xs">
                                ready
                              </Badge>
                            ) : null}
                          </Group>
                        </Table.Td>
                      </Table.Tr>
                    )
                  })}
                </Table.Tbody>
              </Table>
            </ScrollArea.Autosize>

            <Text fz="xs" c="dimmed">
              {chosen.length} selected.
              {hiddenCount > 0
                ? ` ${hiddenCount} detector${hiddenCount === 1 ? '' : 's'} that cannot read ${/^[aeiou]/.test(mediaType) ? 'an' : 'a'} ${mediaType} ${hiddenCount === 1 ? 'is' : 'are'} not listed.`
                : ''}
            </Text>
            {chosenUntrained > 0 ? (
              <Banner color="yellow" title={`${chosenUntrained} research-only detector(s) selected`}>
                Their output is noise. It is shown in the results with a red label and never counts
                toward the summary.
              </Banner>
            ) : null}
            {chosenMissing > 0 ? (
              <Banner color="yellow" title={`${chosenMissing} selected detector(s) have no weights`}>
                Fetch them first with <code>verity weights pull</code>, or they will produce no
                finding.
              </Banner>
            ) : null}
          </Stack>
        )}

        <div>
          <Text fw={500} fz="sm" mb={6}>
            Report
          </Text>
          <SegmentedControl
            value={report}
            onChange={(value) => setReport(value as ReportChoice)}
            data={[
              { value: 'pdf', label: 'PDF', disabled: !canReportPdf },
              { value: 'html', label: 'HTML' },
              { value: 'none', label: 'No report' },
            ]}
          />
          {!canReportPdf ? (
            <Text fz="xs" c="dimmed" mt={4}>
              PDF export is unavailable on this server (WeasyPrint missing).
            </Text>
          ) : null}
        </div>

        <Accordion variant="contained">
          <Accordion.Item value="advanced">
            <Accordion.Control>Advanced: what to extract before detection</Accordion.Control>
            <Accordion.Panel>
              <Checkbox.Group value={extractors} onChange={setExtractors}>
                <Stack gap={6}>
                  {EXTRACTORS.map((extractor) => (
                    <Checkbox
                      key={extractor.key}
                      value={extractor.key}
                      size="sm"
                      label={extractor.label}
                    />
                  ))}
                </Stack>
              </Checkbox.Group>
              <Text fz="xs" c="dimmed" mt="sm">
                Leave all on unless you know a detector does not need an artifact.
              </Text>
            </Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>

      <Group justify="flex-end" mt="lg">
        <Button variant="default" onClick={onClose}>
          Cancel
        </Button>
        <Button onClick={submit} disabled={mode === 'choose' && chosen.length === 0}>
          {mode === 'choose' ? `Run ${chosen.length} detector${chosen.length === 1 ? '' : 's'}` : 'Run'}
        </Button>
      </Group>
    </Modal>
  )
}

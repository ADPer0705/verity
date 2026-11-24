/**
 * Everything one detector run left behind, browsable in place.
 *
 * Explainability images are served as files so an investigator can inspect
 * them at full resolution; the raw JSON and the subprocess logs are shown
 * verbatim because they are what gets quoted when a finding is challenged.
 * Each panel fetches only when opened — a case with thirty findings should
 * not fire ninety requests on load.
 */

import { useMemo } from 'react'
import { Anchor, Badge, Button, CopyButton, Group, Paper, Stack, Tabs, Text } from '@mantine/core'
import { IconCopy, IconExternalLink } from '@tabler/icons-react'

import { api, fileUrl } from '@/api/client'
import { useAsync } from '@/api/hooks'
import type { CoverageFamily, Finding } from '@/api/types'
import { ImageGallery } from '@/components/ImageGallery'
import type { GalleryImage } from '@/components/ImageGallery'
import {
  AuxSignalsTable,
  PerFaceTable,
  RegionEnergyTable,
  unitLabels,
} from '@/components/FindingTables'
import { FamilyChips, SectionTitle, TierBadge } from '@/components/Primitives'
import { CLASSIFICATION_TAG } from '@/lib/verdict'

export function ExplainabilityPanel({ mediaId, finding }: { mediaId: string; finding: Finding }) {
  const assets = useAsync(
    () => api.detectorAssets(mediaId, finding.detector_id),
    [mediaId, finding.detector_id],
  )
  const labels = unitLabels(finding.unit)

  const probabilityByFace = useMemo(() => {
    const map = new Map<string, number>()
    for (const face of finding.per_face) {
      if (face.face_id) map.set(String(face.face_id), face.fake_prob ?? 0)
    }
    return map
  }, [finding.per_face])

  const decorate = (files: { name: string; path: string }[] = [], suffix?: string): GalleryImage[] =>
    [...files]
      .map((file) => {
        const stem = file.name.replace(/\.[^.]+$/, '').replace(suffix ?? '', '')
        const probability = probabilityByFace.get(stem)
        return {
          ...(file as GalleryImage),
          caption: stem,
          badge: probability != null ? `P=${probability.toFixed(3)}` : undefined,
          _prob: probability ?? -1,
        }
      })
      .sort((a, b) => b._prob - a._prob || a.name.localeCompare(b.name))

  if (assets.loading) {
    return (
      <Text c="dimmed" fz="sm">
        Loading explainability artifacts…
      </Text>
    )
  }

  if (assets.error) {
    return (
      <Text c="red" fz="sm">
        {assets.error}
      </Text>
    )
  }

  if (!assets.data?.exists) {
    return (
      <Text c="dimmed" fz="sm">
        No output directory found for this detector run. Artifacts are written to{' '}
        <span className="vy-mono">
          workspace/output/{mediaId}_{finding.detector_id}/
        </span>
        .
      </Text>
    )
  }

  const fullFrames = decorate(assets.data.full_frames, '_frame')
  const faces = decorate(assets.data.faces)
  const frames = decorate(assets.data.frames)
  const attention = decorate(assets.data.attention_maps, '_attention')
  const reconstructions = decorate(assets.data.reconstructions, '_recon')
  const hasAux = Object.keys(finding.aux_aggregate ?? {}).length > 0
  const hasRegions = Object.keys(finding.region_aggregate ?? {}).length > 0

  const nothing =
    !assets.data.timeline &&
    !fullFrames.length &&
    !faces.length &&
    !frames.length &&
    !attention.length &&
    !reconstructions.length &&
    !hasAux &&
    !hasRegions

  const threshold = finding.stats?.threshold ?? 0.5

  return (
    <Stack gap="lg">
      {assets.data.output_dir ? (
        <Group justify="space-between" wrap="wrap">
          <Text fz="xs" c="dimmed" ff="monospace" className="vy-break">
            {assets.data.output_dir}
          </Text>
          {finding.model?.architecture ? (
            <Badge variant="outline" color="gray" size="xs">
              {finding.model.architecture}
            </Badge>
          ) : null}
        </Group>
      ) : null}

      {nothing ? (
        <Text c="dimmed" fz="sm">
          The output directory exists but holds no visual explainability artifacts for this run.
        </Text>
      ) : null}

      {hasAux ? <AuxSignalsTable finding={finding} /> : null}

      {hasRegions ? <RegionEnergyTable finding={finding} /> : null}

      {assets.data.timeline ? (
        <div>
          <SectionTitle>
            {finding.unit === 'window'
              ? 'Audio / AV window prediction timeline'
              : 'Temporal prediction timeline'}
          </SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Per-{labels.singular} fake probability across the media (model decision midpoint{' '}
            {threshold.toFixed(2)}; high-confidence manipulation &gt; 0.70, authentic &lt; 0.30).
          </Text>
          <Anchor href={fileUrl(mediaId, assets.data.timeline.path)} target="_blank" rel="noreferrer">
            <img
              src={fileUrl(mediaId, assets.data.timeline.path)}
              alt="Prediction timeline"
              style={{
                width: '100%',
                background: 'var(--mantine-color-default-hover)',
                border: '1px solid var(--mantine-color-default-border)',
                borderRadius: 'var(--mantine-radius-sm)',
              }}
            />
          </Anchor>
        </div>
      ) : null}

      {fullFrames.length ? (
        <div>
          <SectionTitle>Top suspicious video frames ({fullFrames.length})</SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Highest-scoring full video frames annotated with the detected face bounding box and fake
            probability.
          </Text>
          <ImageGallery mediaId={mediaId} images={fullFrames} layout="wide" />
        </div>
      ) : null}

      {attention.length ? (
        <div>
          <SectionTitle>Spatial activation & attention maps ({attention.length})</SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Original crop → spatial activation heatmap → overlay showing the exact regions that
            drove the detector's score.
          </Text>
          <ImageGallery mediaId={mediaId} images={attention} layout="maps" />
        </div>
      ) : null}

      {reconstructions.length ? (
        <div>
          <SectionTitle>Autoencoder reconstructions & residuals ({reconstructions.length})</SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Original → reconstruction → pixel residual |Δ|.
          </Text>
          <ImageGallery mediaId={mediaId} images={reconstructions} layout="wide" />
        </div>
      ) : null}

      {faces.length ? (
        <div>
          <SectionTitle>Scored face crops ({faces.length})</SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Aligned face crops evaluated by the detector, sorted by fake probability (highest first).
          </Text>
          <ImageGallery mediaId={mediaId} images={faces} layout="faces" />
        </div>
      ) : null}

      {frames.length ? (
        <div>
          <SectionTitle>
            {finding.unit === 'window'
              ? `Scored AV window keyframes (${frames.length})`
              : `Scored full frames (${frames.length})`}
          </SectionTitle>
          <Text fz="xs" c="dimmed" mb="xs">
            Whole-frame thumbnails evaluated by the detector, sorted by fake probability.
          </Text>
          <ImageGallery mediaId={mediaId} images={frames} layout="maps" />
        </div>
      ) : null}
    </Stack>
  )
}

export function RawPanel({ mediaId, finding }: { mediaId: string; finding: Finding }) {
  const raw = useAsync(
    () => api.detectorRaw(mediaId, finding.detector_id).catch(() => ({})),
    [mediaId, finding.detector_id],
  )
  const text = JSON.stringify(raw.data ?? {}, null, 2)

  return (
    <div>
      <Group justify="space-between" mb="xs">
        <Text fz="xs" c="dimmed">
          Verbatim <span className="vy-mono">result.json</span> from the detector and the parsed{' '}
          <span className="vy-mono">finding.json</span> Verity stored.
        </Text>
        <CopyButton value={text}>
          {({ copied, copy }) => (
            <Button
              variant="subtle"
              size="compact-xs"
              color={copied ? 'teal' : 'gray'}
              leftSection={<IconCopy size={12} />}
              onClick={copy}
            >
              {copied ? 'Copied' : 'Copy JSON'}
            </Button>
          )}
        </CopyButton>
      </Group>
      {raw.loading ? (
        <Text c="dimmed" fz="sm">
          Loading…
        </Text>
      ) : (
        <pre className="vy-pre">{text}</pre>
      )}
    </div>
  )
}

export function LogsPanel({ mediaId, finding }: { mediaId: string; finding: Finding }) {
  const logs = useAsync(
    () => api.detectorLogs(mediaId, finding.detector_id),
    [mediaId, finding.detector_id],
  )

  if (logs.loading) {
    return (
      <Text c="dimmed" fz="sm">
        Loading…
      </Text>
    )
  }
  if (logs.error) {
    return (
      <Text c="red" fz="sm">
        {logs.error}
      </Text>
    )
  }

  return (
    <Stack gap="md">
      <div>
        <SectionTitle>stderr</SectionTitle>
        <pre className="vy-pre">{logs.data?.stderr || '(empty)'}</pre>
      </div>
      <div>
        <SectionTitle>stdout</SectionTitle>
        <pre className="vy-pre">{logs.data?.stdout || '(empty)'}</pre>
      </div>
    </Stack>
  )
}

/** Header strip + tabs for one detector's run. */
export function DetectorEvidence({
  mediaId,
  finding,
  families,
  onOpenFinding,
}: {
  mediaId: string
  finding: Finding
  families?: CoverageFamily[] | null
  onOpenFinding?: () => void
}) {
  const labels = unitLabels(finding.unit)

  return (
    <div>
      <Group justify="space-between" align="flex-start" mb="sm" wrap="wrap">
        <div>
          <Group gap="xs" wrap="wrap">
            <Text fw={600} fz="md">
              {finding.detector_name}
            </Text>
            <Badge color={CLASSIFICATION_TAG[finding.classification]} size="sm">
              {finding.classification}
            </Badge>
            {!finding.trained ? <TierBadge tier={finding.tier} /> : null}
            {finding.input_mode ? (
              <Badge variant="outline" color="gray" size="xs" ff="monospace">
                {finding.input_mode}
              </Badge>
            ) : null}
          </Group>
          <Text fz="xs" c="dimmed" ff="monospace" mt={2}>
            {finding.detector_id} v{finding.detector_version} · mean P(fake){' '}
            {finding.mean_fake_probability.toFixed(4)} · confidence {finding.confidence.toFixed(4)}
            {finding.device ? ` · ${finding.device}` : ''}
            {finding.timing_seconds?.total != null ? ` · ${finding.timing_seconds.total}s` : ''}
          </Text>
          <div style={{ marginTop: 6 }}>
            <FamilyChips targets={finding.targets} families={families} />
          </div>
        </div>
        {onOpenFinding ? (
          <Button
            variant="subtle"
            size="compact-xs"
            rightSection={<IconExternalLink size={12} />}
            onClick={onOpenFinding}
          >
            Reasoning in Findings
          </Button>
        ) : null}
      </Group>

      {finding.reasoning.summary ? (
        <Paper withBorder p="xs" mb="sm" bg="var(--mantine-color-default-hover)">
          <Text fz="xs" lh={1.5}>
            {finding.reasoning.summary}
          </Text>
        </Paper>
      ) : null}

      <Tabs defaultValue="explainability" keepMounted={false} variant="outline">
        <Tabs.List>
          <Tabs.Tab value="explainability">Explainability & visuals</Tabs.Tab>
          <Tabs.Tab value="faces">
            Per-{labels.singular} ({finding.per_face.length})
          </Tabs.Tab>
          <Tabs.Tab value="raw">Raw JSON</Tabs.Tab>
          <Tabs.Tab value="logs" disabled={!finding.has_logs && !finding.output_dir}>
            Logs
          </Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="explainability" pt="md">
          <ExplainabilityPanel mediaId={mediaId} finding={finding} />
        </Tabs.Panel>
        <Tabs.Panel value="faces" pt="md">
          <PerFaceTable
            rows={finding.per_face}
            unit={finding.unit}
            inputMode={finding.input_mode}
          />
        </Tabs.Panel>
        <Tabs.Panel value="raw" pt="md">
          <RawPanel mediaId={mediaId} finding={finding} />
        </Tabs.Panel>
        <Tabs.Panel value="logs" pt="md">
          <LogsPanel mediaId={mediaId} finding={finding} />
        </Tabs.Panel>
      </Tabs>
    </div>
  )
}

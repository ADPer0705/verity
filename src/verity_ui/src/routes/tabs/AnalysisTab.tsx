/**
 * Media characterization — what the selector's decisions were based on.
 *
 * The per-frame face histogram is the part worth reading closely: it shows
 * exactly which sampled frames contained faces, which is what determines
 * whether a face-based detector had anything to work with.
 */

import { Button, CopyButton, Grid, Group, Text } from '@mantine/core'
import { IconChartHistogram, IconCopy } from '@tabler/icons-react'

import type { CaseDetail } from '@/api/types'
import { EmptyState, KeyValue, SectionTitle } from '@/components/Primitives'
import { formatDuration } from '@/lib/format'

function FaceHistogram({
  frames,
}: {
  frames: { frame: number; face_count: number; max_face_size: number }[]
}) {
  if (!frames.length) {
    return (
      <Text c="dimmed" fz="sm">
        No sampled frame contained a detectable face. Face-based detectors are excluded on that
        basis — see the Selection tab.
      </Text>
    )
  }

  const peak = Math.max(...frames.map((frame) => frame.face_count))

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 2,
          height: '8rem',
          padding: '0.5rem',
          background: 'var(--mantine-color-default)',
          border: '1px solid var(--mantine-color-default-border)',
          borderRadius: 'var(--mantine-radius-sm)',
          overflowX: 'auto',
        }}
        role="img"
        aria-label={`Face counts across ${frames.length} sampled frames, peaking at ${peak}.`}
      >
        {frames.map((frame) => (
          <div
            key={frame.frame}
            title={`Frame ${frame.frame}: ${frame.face_count} face(s), largest ${frame.max_face_size}px`}
            style={{
              flex: '1 0 6px',
              minWidth: 6,
              height: `${Math.max(4, (frame.face_count / peak) * 100)}%`,
              background: 'var(--mantine-color-indigo-5)',
              borderRadius: 1,
            }}
          />
        ))}
      </div>
      <Text fz="xs" c="dimmed" mt="xs">
        {frames.length} sampled frame{frames.length === 1 ? '' : 's'} contained faces · peak {peak}{' '}
        face{peak === 1 ? '' : 's'} in a single frame. Hover a bar for the frame index and largest
        face size.
      </Text>
    </div>
  )
}

export function AnalysisTab({ detail }: { detail: CaseDetail }) {
  const { analysis } = detail

  if (!analysis) {
    return (
      <EmptyState
        icon={<IconChartHistogram size={26} />}
        title="This media has not been characterized yet"
        body="Run Analyze (or a full examination) to measure the file with ffprobe and scan sampled frames for faces. Detector eligibility is decided from these values."
      />
    )
  }

  const media = analysis.media
  const json = JSON.stringify(analysis, null, 2)

  return (
    <Grid gutter="xl">
      <Grid.Col span={{ base: 12, md: 6 }}>
        <SectionTitle>Characteristics</SectionTitle>
        <KeyValue
          rows={[
            ['Media type', media.media_type],
            ['Container format', media.format || '—'],
            ['Duration', formatDuration(media.duration_seconds)],
            ['Resolution', media.width && media.height ? `${media.width} × ${media.height}` : '—'],
            ['Frame rate', media.fps ? `${media.fps} fps` : '—'],
            ['Frame count', media.frame_count?.toLocaleString() ?? '—'],
            ['Video codec', media.video_codec || '—'],
            ['Audio track', media.has_audio_track ? media.audio_codec || 'present' : 'none'],
            ['Faces present', media.face_present ? 'yes' : 'no'],
            ['Total faces', media.face_count.toLocaleString()],
            ['Sampled frames', media.sampled_frames.toLocaleString()],
            ['Largest face', media.max_face_size ? `${media.max_face_size} px` : '—'],
          ]}
        />
        <Text fz="xs" c="dimmed" mt="xs">
          Blank video fields mean ffprobe was unavailable or could not read the container. Face
          figures come from OpenCV's Haar cascade over evenly sampled frames and are deterministic
          for the same file.
        </Text>
      </Grid.Col>

      <Grid.Col span={{ base: 12, md: 6 }}>
        <SectionTitle>Faces per sampled frame</SectionTitle>
        <FaceHistogram frames={analysis.faces_per_frame} />

        <Group justify="space-between" mt="xl" mb={0}>
          <SectionTitle>Raw analysis</SectionTitle>
          <CopyButton value={json}>
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
        <pre className="vy-pre">{json}</pre>
      </Grid.Col>
    </Grid>
  )
}

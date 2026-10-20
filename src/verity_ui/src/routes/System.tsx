/**
 * What this installation is and whether it can do its job.
 *
 * Read-only on purpose: configuration comes from `config/pipeline.yaml` and
 * the `VERITY_*` environment variables (set in docker-compose.yml or `.env`),
 * so it survives restarts and is the same for every user of the server.
 */

import type { ReactNode } from 'react'
import { Badge, Code, Grid, Group, Stack, Text } from '@mantine/core'
import { IconCircleCheck, IconCircleX } from '@tabler/icons-react'

import { api } from '@/api/client'
import { useAsync } from '@/api/hooks'
import {
  Banner,
  ErrorBanner,
  KeyValue,
  LoadingBlock,
  PageHeader,
  SectionTitle,
} from '@/components/Primitives'

function Availability({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <Group gap={6} wrap="nowrap" align="flex-start">
      {ok ? (
        <IconCircleCheck size={16} color="var(--mantine-color-teal-6)" style={{ flexShrink: 0 }} />
      ) : (
        <IconCircleX size={16} color="var(--mantine-color-red-6)" style={{ flexShrink: 0 }} />
      )}
      <span>{children}</span>
    </Group>
  )
}

export default function System() {
  const status = useAsync(() => api.system(), [])


  if (status.loading && !status.data) {
    return (
      <div className="vy-page">
        <LoadingBlock lines={6} />
      </div>
    )
  }

  const environment = status.data?.environment
  const readiness = status.data?.readiness

  return (
    <div className="vy-page">
      <PageHeader
        title="System"
        subtitle="This installation's configuration and capabilities. Change configuration in config/pipeline.yaml or with VERITY_* environment variables, then restart the server."
      />

      <ErrorBanner error={status.error} onRetry={status.reload} />

      <Grid gutter="xl">
        <Grid.Col span={{ base: 12, md: 7 }}>
          <SectionTitle>Configuration</SectionTitle>
          {status.data ? (
            <KeyValue
              rows={[
                ['Detectors', <span key="d" className="vy-mono vy-break">{status.data.settings.detectors_dir}</span>],
                ['Workspace', <span key="w" className="vy-mono vy-break">{status.data.settings.workspace}</span>],
                ['Reports', <span key="o" className="vy-mono vy-break">{status.data.settings.output_dir}</span>],
                ['Extractors', status.data.settings.extractors.join(', ')],
                [
                  'Downloads',
                  status.data.settings.offline
                    ? 'Offline: only weights already on disk are used'
                    : 'Allowed: missing model weights may be fetched at first run',
                ],
              ]}
            />
          ) : null}
          <Text fz="xs" c="dimmed" mt="sm">
            Set with <Code>VERITY_DETECTORS_DIR</Code>, <Code>VERITY_WORKSPACE</Code>,{' '}
            <Code>VERITY_OUTPUT_DIR</Code> and <Code>VERITY_OFFLINE=1</Code>. See{' '}
            <Code>.env.example</Code> for the full list.
          </Text>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 5 }}>
          <SectionTitle>Environment</SectionTitle>
          {environment ? (
            <KeyValue
              rows={[
                [
                  'Detectors',
                  <Availability key="d" ok={environment.detector_count > 0}>
                    {environment.detector_count} definition
                    {environment.detector_count === 1 ? '' : 's'} found
                  </Availability>,
                ],
                [
                  'Detector runtime',
                  <Availability key="r" ok={Boolean(environment.detector_runner)}>
                    <span className="vy-mono vy-break">
                      {environment.detector_runner ?? 'not found'}
                    </span>
                  </Availability>,
                ],
                [
                  'uv',
                  <Availability key="u" ok={Boolean(environment.uv)}>
                    {environment.uv ?? 'not on PATH — provisioning unavailable'}
                  </Availability>,
                ],
                [
                  'ffprobe',
                  <Availability key="f" ok={Boolean(environment.ffprobe)}>
                    {environment.ffprobe ?? 'not on PATH — duration, resolution and FPS unavailable'}
                  </Availability>,
                ],
                [
                  'OpenCV',
                  <Availability key="c" ok={environment.opencv}>
                    {environment.opencv ? 'available — face scanning enabled' : 'missing'}
                  </Availability>,
                ],
                [
                  'WeasyPrint',
                  <Availability key="w" ok={environment.weasyprint}>
                    {environment.weasyprint
                      ? 'available — PDF export enabled'
                      : (environment.weasyprint_error ?? 'missing — HTML reports only')}
                  </Availability>,
                ],
                [
                  'Hardware profile',
                  <Badge key="h" variant="outline" color="gray">
                    {environment.hw_profile}
                  </Badge>,
                ],
              ]}
            />
          ) : null}

          {readiness?.blockers.length ? (
            <Stack gap="xs" mt="md">
              {readiness.blockers.map((blocker) => (
                <Banner
                  key={`${blocker.capability}-${blocker.message}`}
                  color="yellow"
                  title={blocker.capability}
                >
                  {blocker.message}
                </Banner>
              ))}
            </Stack>
          ) : (
            <Banner color="teal" title="Fully operational" mt="md">
              Every pipeline capability is available in this environment.
            </Banner>
          )}
        </Grid.Col>
      </Grid>
    </div>
  )
}

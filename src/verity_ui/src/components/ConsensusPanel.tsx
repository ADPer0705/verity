/**
 * The case summary: a plain-language headline, what it means, the
 * agreement bar (the honest picture of a panel of detectors that may
 * disagree) and the technical aggregate for those who need it.
 */

import { ActionIcon, Badge, Group, Progress, Text, Tooltip } from '@mantine/core'
import { IconInfoCircle } from '@tabler/icons-react'

import type { Consensus } from '@/api/types'
import { presentBand, reliability } from '@/lib/verdict'

const SEGMENT_COLOR = { fake: 'red', real: 'teal', uncertain: 'violet' } as const

export function AgreementBar({ consensus }: { consensus: Consensus }) {
  const { fake, real, uncertain, total } = consensus
  if (total === 0) return null

  const segments = [
    { key: 'fake', count: fake, label: 'Signs of manipulation' },
    { key: 'real', count: real, label: 'No signs of manipulation' },
    { key: 'uncertain', count: uncertain, label: 'Undecided' },
  ].filter((segment) => segment.count > 0) as {
    key: keyof typeof SEGMENT_COLOR
    count: number
    label: string
  }[]

  return (
    <div>
      <Progress.Root
        size={10}
        aria-label={`${fake} of ${total} methods reported manipulation, ${real} reported no manipulation, ${uncertain} were undecided.`}
      >
        {segments.map((segment) => (
          <Progress.Section
            key={segment.key}
            value={(segment.count / total) * 100}
            color={SEGMENT_COLOR[segment.key]}
          />
        ))}
      </Progress.Root>
      <Group gap="md" mt={6}>
        {segments.map((segment) => (
          <Group gap={6} key={segment.key}>
            <span
              aria-hidden
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: `var(--mantine-color-${SEGMENT_COLOR[segment.key]}-6)`,
              }}
            />
            <Text fz="xs" c="dimmed">
              {segment.count} × {segment.label}
            </Text>
          </Group>
        ))}
      </Group>
    </div>
  )
}

export function ConsensusVerdict({ consensus }: { consensus: Consensus }) {
  const presentation = presentBand(consensus.band)
  const trust = reliability(consensus)

  return (
    <section className={`vy-verdict vy-verdict--${presentation.tone}`}>
      <Group gap="xs">
        <Badge color={presentation.tagType} variant="filled">
          {presentation.label}
        </Badge>
        {consensus.total > 0 ? (
          <Badge variant="outline" color="gray">
            {trust.label}
          </Badge>
        ) : null}
        {consensus.disagreement ? (
          <Tooltip label={trust.explanation} multiline w={320} position="bottom">
            <ActionIcon variant="subtle" color="gray" size="sm" aria-label="Why methods disagree">
              <IconInfoCircle size={16} />
            </ActionIcon>
          </Tooltip>
        ) : null}
      </Group>

      <h2 className="vy-verdict__headline">{presentation.plainHeadline}</h2>

      <p className="vy-verdict__body">{presentation.plainBody}</p>

      {consensus.total > 0 ? (
        <div style={{ marginTop: '1rem' }}>
          <AgreementBar consensus={consensus} />
        </div>
      ) : null}

      {consensus.total > 0 ? (
        <Text fz="xs" c="dimmed" mt="md">
          Mean fake probability across detectors{' '}
          <Text component="span" ff="monospace">
            {consensus.mean_fake_probability?.toFixed(4) ?? '—'}
          </Text>{' '}
          · agreement{' '}
          <Text component="span" ff="monospace">
            {Math.round(consensus.agreement * 100)}%
          </Text>
          . {presentation.technicalSummary} This aggregate is for orientation only; the
          per-detector findings are the record.
          {consensus.untrained_excluded ? (
            <>
              {' '}
              {consensus.untrained_excluded} untrained-head finding
              {consensus.untrained_excluded === 1 ? '' : 's'} excluded from the vote.
            </>
          ) : null}
        </Text>
      ) : null}
    </section>
  )
}

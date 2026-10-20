/**
 * Handling evidence — preserving and reporting a suspect file.
 *
 * Procedural guidance only: what to keep, what not to do, and where to
 * report. It deliberately does not name offences or cite provisions — that
 * mapping is the investigator's and the legal reviewer's call, and a wrong
 * citation on a public-facing screen would be worse than none.
 */

import { Link, useParams } from 'react-router-dom'
import { Button, Group, List, Paper, SimpleGrid, Stack, Text, Title } from '@mantine/core'
import { IconArrowLeft, IconDownload, IconExternalLink, IconPhone } from '@tabler/icons-react'

import { api, mediaFileUrl, reportUrl } from '@/api/client'
import { useAsync } from '@/api/hooks'
import { Banner, CopyValue } from '@/components/Primitives'
import { shortHash } from '@/lib/format'

function Label({ children }: { children: string }) {
  return (
    <Text fz="xs" c="dimmed" fw={600} tt="uppercase" style={{ letterSpacing: '0.05em' }}>
      {children}
    </Text>
  )
}

export default function CitizenGuidance() {
  const { mediaId = '' } = useParams()
  const { data } = useAsync(() => (mediaId ? api.case(mediaId) : Promise.resolve(null)), [mediaId])

  return (
    <div className="vy-page--narrow">
      {mediaId ? (
        <Button
          variant="subtle"
          color="gray"
          size="compact-sm"
          leftSection={<IconArrowLeft size={14} />}
          component={Link}
          to={`/case/${mediaId}`}
          mb="sm"
        >
          Back to the case
        </Button>
      ) : null}

      <Title order={1} fz="2rem" fw={500}>
        Preserving and reporting this file
      </Title>
      <Text c="dimmed" fz="md" lh={1.6} mt="xs" maw="42rem">
        If this file is being used to harass, impersonate, extort or defraud, what you do in the
        next few hours matters. Evidence is lost easily and it is rarely recoverable.
      </Text>

      <Banner color="blue" title="This is procedural guidance, not legal advice" mt="md">
        Which offences apply to your situation is a decision for the investigating officer and a
        legal professional, based on the full facts.
      </Banner>

      {/* ---------------- do now ---------------- */}

      <Title order={2} mt="xl" mb="sm">
        Do these first
      </Title>
      <List type="ordered" fz="sm" spacing="sm">
        <List.Item>
          <strong>Keep the original.</strong> Do not delete, crop, re-save, re-upload or forward it
          to more people. Every re-share strips information an examiner needs.
        </List.Item>
        <List.Item>
          <strong>Capture where it came from.</strong> Screenshot the message, post or email
          showing the sender's name or handle, the account profile, and the date and time — with
          the surrounding conversation visible.
        </List.Item>
        <List.Item>
          <strong>Write down the timeline.</strong> When you first saw it, who sent it, what they
          demanded, and every contact since. Do this while it is fresh.
        </List.Item>
        <List.Item>
          <strong>Do not pay and do not negotiate</strong> if money or more images are being
          demanded. Payment does not end the demands.
        </List.Item>
        <List.Item>
          <strong>Report it.</strong> Use the channels below. Bring the original file, your
          screenshots and your timeline.
        </List.Item>
      </List>

      {/* ---------------- avoid ---------------- */}

      <Title order={2} mt="xl" mb="sm">
        Avoid these
      </Title>
      <List fz="sm" spacing="sm">
        <List.Item>
          Editing the file to "prove" it is fake — this destroys the very traces an examination
          relies on.
        </List.Item>
        <List.Item>
          Confronting the sender before you have reported it, which often prompts them to delete
          accounts and evidence.
        </List.Item>
        <List.Item>
          Relying on the automated result alone. It is a signal for an investigator, not a finding
          a platform or court will act on by itself.
        </List.Item>
      </List>

      {/* ---------------- where ---------------- */}

      <Title order={2} mt="xl" mb="sm">
        Where to report, in India
      </Title>
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
        <Paper withBorder p="md">
          <Title order={3}>National Cyber Crime Reporting Portal</Title>
          <Text c="dimmed" fz="sm" mt="xs" mb="sm" lh={1.5}>
            The official channel for cybercrime complaints, including a dedicated route for crimes
            against women and children that allows anonymous reporting.
          </Text>
          <Button
            variant="default"
            size="xs"
            rightSection={<IconExternalLink size={14} />}
            component="a"
            href="https://cybercrime.gov.in"
            target="_blank"
            rel="noreferrer noopener"
          >
            cybercrime.gov.in
          </Button>
        </Paper>
        <Paper withBorder p="md">
          <Title order={3}>Cyber crime helpline — 1930</Title>
          <Text c="dimmed" fz="sm" mt="xs" mb="sm" lh={1.5}>
            For financial fraud, call as soon as possible. The first hours are what decide whether
            a transfer can still be stopped.
          </Text>
          <Button
            variant="default"
            size="xs"
            leftSection={<IconPhone size={14} />}
            component="a"
            href="tel:1930"
          >
            Call 1930
          </Button>
        </Paper>
      </SimpleGrid>
      <Text fz="xs" c="dimmed" mt="sm">
        You can also go to your nearest police station in person. They cannot refuse to record a
        complaint on the grounds that the offence happened online.
      </Text>

      {/* ---------------- what to bring ---------------- */}

      {data ? (
        <>
          <Title order={2} mt="xl" mb={4}>
            What to take with you
          </Title>
          <Text c="dimmed" fz="sm" mb="sm" maw="42rem">
            Verity fingerprinted your file when it was recorded. Quoting that fingerprint lets an
            examiner confirm the copy you hand over is the same one that was examined here.
          </Text>
          <Paper withBorder p="md">
            <Stack gap="sm">
              <div>
                <Label>File</Label>
                <Text fz="sm">{data.media.file_name}</Text>
              </div>
              <div>
                <Label>Reference (case id)</Label>
                <CopyValue value={data.media_id} label="case id" />
              </div>
              <div>
                <Label>SHA-256 fingerprint</Label>
                <CopyValue
                  value={data.media.checksum}
                  label="checksum"
                  truncate={shortHash(data.media.checksum, 16, 12)}
                />
              </div>
            </Stack>
            <Group gap="sm" mt="md">
              {data.reports.pdf ? (
                <Button
                  size="xs"
                  leftSection={<IconDownload size={14} />}
                  component="a"
                  href={reportUrl(mediaId, 'pdf', true)}
                >
                  Download the examination report
                </Button>
              ) : null}
              <Button
                size="xs"
                variant="default"
                leftSection={<IconDownload size={14} />}
                component="a"
                href={mediaFileUrl(mediaId, true)}
              >
                Download the recorded copy
              </Button>
            </Group>
          </Paper>
        </>
      ) : null}

      <Group mt="xl">
        <Button variant="subtle" color="gray" component={Link} to="/">
          Back to cases
        </Button>
      </Group>
    </div>
  )
}

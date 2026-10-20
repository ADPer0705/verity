/**
 * Report generation and preview.
 *
 * The HTML report is rendered inline in a sandboxed iframe so an examiner can
 * read exactly what will be exported — the PDF is produced from the same
 * string, so what is previewed is what is filed.
 */

import { Badge, Button, Group, SimpleGrid, Text } from '@mantine/core'
import { IconExternalLink, IconFileText, IconFileTypePdf, IconRefresh } from '@tabler/icons-react'

import { reportUrl } from '@/api/client'
import type { CaseDetail } from '@/api/types'
import { Banner, EmptyState, KeyValue, SectionTitle } from '@/components/Primitives'
import { formatBytes, formatTimestamp } from '@/lib/format'

export function ReportTab({
  detail,
  canReportPdf,
  pdfBlocker,
  onGenerate,
  busy,
}: {
  detail: CaseDetail
  canReportPdf: boolean
  pdfBlocker?: string
  onGenerate: (format: 'pdf' | 'html') => void
  busy: boolean
}) {
  const { reports } = detail
  const hasAny = Boolean(reports.html || reports.pdf)

  return (
    <div>
      <Group justify="space-between" align="flex-start" mb="md" wrap="wrap">
        <div>
          <SectionTitle>Examination report</SectionTitle>
          <Text fz="sm" c="dimmed" maw="46rem">
            Deterministic: the same media and detectors always produce the same report. It states
            findings and their provenance — it does not render a verdict.
          </Text>
        </div>
        <Group gap="xs">
          <Button
            variant={hasAny ? 'default' : 'filled'}
            leftSection={<IconRefresh size={16} />}
            disabled={busy}
            onClick={() => onGenerate('html')}
          >
            {busy ? 'Generating…' : 'Generate HTML'}
          </Button>
          <Button
            leftSection={<IconFileTypePdf size={16} />}
            disabled={busy || !canReportPdf}
            onClick={() => onGenerate('pdf')}
          >
            Generate PDF
          </Button>
        </Group>
      </Group>

      {!canReportPdf ? (
        <Banner color="yellow" title="PDF export unavailable" mb="md">
          {pdfBlocker ?? 'WeasyPrint is not available in this environment.'} HTML reports work and
          can be printed to PDF from the browser.
        </Banner>
      ) : null}

      {detail.findings.length === 0 ? (
        <Banner color="blue" title="No findings to report" mb="md">
          A report generated now will contain the media identity and the selection record, but no
          detector findings.
        </Banner>
      ) : null}

      {!hasAny ? (
        <EmptyState
          icon={<IconFileText size={26} />}
          title="No report generated yet"
          body="Generate one to produce the media identity block, the findings summary, the excluded-detector table and one section per finding with its reasoning, key figures and evidence images."
        />
      ) : (
        <>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg" mb="md">
            {reports.html ? (
              <KeyValue
                rows={[
                  ['Format', <Badge key="t" color="indigo" size="sm">HTML</Badge>],
                  ['Generated', formatTimestamp(reports.html.generated_at)],
                  ['Size', formatBytes(reports.html.size_bytes)],
                  ['Path', <span key="p" className="vy-mono vy-break">{reports.html.path}</span>],
                ]}
              />
            ) : null}
            {reports.pdf ? (
              <KeyValue
                rows={[
                  ['Format', <Badge key="t" color="pink" size="sm">PDF</Badge>],
                  ['Generated', formatTimestamp(reports.pdf.generated_at)],
                  ['Size', formatBytes(reports.pdf.size_bytes)],
                  ['Path', <span key="p" className="vy-mono vy-break">{reports.pdf.path}</span>],
                ]}
              />
            ) : null}
          </SimpleGrid>

          <Group gap="xs" mb="md">
            {reports.html ? (
              <>
                <Button
                  variant="default"
                  size="xs"
                  rightSection={<IconExternalLink size={14} />}
                  component="a"
                  href={reportUrl(detail.media_id, 'html')}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open HTML in a new tab
                </Button>
                <Button
                  variant="subtle"
                  size="xs"
                  component="a"
                  href={reportUrl(detail.media_id, 'html', true)}
                >
                  Download HTML
                </Button>
              </>
            ) : null}
            {reports.pdf ? (
              <Button
                variant="subtle"
                size="xs"
                leftSection={<IconFileTypePdf size={14} />}
                component="a"
                href={reportUrl(detail.media_id, 'pdf', true)}
              >
                Download PDF
              </Button>
            ) : null}
          </Group>

          {reports.html ? (
            <>
              <SectionTitle>Preview</SectionTitle>
              <iframe
                title="Examination report preview"
                src={reportUrl(detail.media_id, 'html')}
                sandbox=""
                className="vy-report-frame"
              />
            </>
          ) : null}
        </>
      )}
    </div>
  )
}

/**
 * Evidence image grid with a zoom modal.
 *
 * The report embeds these as base64 at fixed widths; here they are served as
 * files so an investigator can actually inspect them at full resolution —
 * which is the difference between reading a report and verifying one.
 */

import { useState } from 'react'
import { Button, Group, Modal, Text } from '@mantine/core'
import { IconDownload } from '@tabler/icons-react'

import { fileUrl } from '@/api/client'
import type { FileEntry } from '@/api/types'

export interface GalleryImage extends FileEntry {
  caption?: string
  badge?: string
}

const PAGE = 48

export function ImageGallery({
  mediaId,
  images,
  layout = 'faces',
  emptyMessage = 'No images were produced for this section.',
}: {
  mediaId: string
  images: GalleryImage[]
  layout?: 'faces' | 'maps' | 'wide'
  emptyMessage?: string
}) {
  const [active, setActive] = useState<GalleryImage | null>(null)
  // A video run can leave several hundred crops; page them so the tab stays usable.
  const [limit, setLimit] = useState(PAGE)
  const shown = images.slice(0, limit)

  if (!images.length) {
    return (
      <Text c="dimmed" fz="sm">
        {emptyMessage}
      </Text>
    )
  }

  return (
    <>
      <div className={`vy-gallery vy-gallery--${layout}`}>
        {shown.map((image) => (
          <button
            type="button"
            key={image.path}
            className="vy-thumb"
            onClick={() => setActive(image)}
            aria-label={`Enlarge ${image.caption ?? image.name}`}
          >
            <img src={fileUrl(mediaId, image.path)} alt={image.caption ?? image.name} loading="lazy" />
            <span className="vy-thumb__caption">
              <span className="vy-break">{image.caption ?? image.name.replace(/\.[^.]+$/, '')}</span>
              {image.badge ? <span>{image.badge}</span> : null}
            </span>
          </button>
        ))}
      </div>
      {limit < images.length ? (
        <Group gap="xs" mt="xs">
          <Button variant="default" size="compact-sm" onClick={() => setLimit((n) => n + PAGE * 2)}>
            Show more ({images.length - limit} remaining)
          </Button>
          <Button variant="subtle" size="compact-sm" onClick={() => setLimit(images.length)}>
            Show all {images.length}
          </Button>
        </Group>
      ) : null}

      <Modal
        opened={active !== null}
        onClose={() => setActive(null)}
        size="xl"
        title={active?.caption ?? active?.name ?? 'Evidence artifact'}
        centered
      >
        {active ? (
          <div className="vy-lightbox">
            <img src={fileUrl(mediaId, active.path)} alt={active.caption ?? active.name} />
            <Group justify="space-between" mt="sm" wrap="wrap">
              <Text ff="monospace" fz="xs" c="dimmed" className="vy-break">
                {active.path}
              </Text>
              <Button
                variant="subtle"
                size="compact-sm"
                leftSection={<IconDownload size={14} />}
                component="a"
                href={fileUrl(mediaId, active.path)}
                download={active.name}
              >
                Download
              </Button>
            </Group>
          </div>
        ) : null}
      </Modal>
    </>
  )
}

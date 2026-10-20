/**
 * Mantine theme shared by both profiles.
 *
 * Calm and dense: one accent colour, small radii, no gradients. Inter is
 * bundled (see main.tsx) so the console renders identically offline; ids,
 * hashes and paths use the platform monospace stack.
 */

import { Badge, Code, Table, createTheme } from '@mantine/core'

export const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

export const theme = createTheme({
  fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  fontFamilyMonospace: MONO,
  headings: {
    fontFamily: 'Inter, system-ui, sans-serif',
    fontWeight: '500',
    sizes: {
      h1: { fontSize: '1.75rem', lineHeight: '1.25' },
      h2: { fontSize: '1.25rem', lineHeight: '1.3' },
      h3: { fontSize: '1rem', lineHeight: '1.4' },
      h4: { fontSize: '0.9375rem', lineHeight: '1.4' },
    },
  },
  primaryColor: 'indigo',
  primaryShade: { light: 6, dark: 5 },
  defaultRadius: 'sm',
  cursorType: 'pointer',
  components: {
    Table: Table.extend({
      defaultProps: { verticalSpacing: 6, horizontalSpacing: 'sm', fz: 'sm' },
    }),
    Badge: Badge.extend({
      defaultProps: { variant: 'light', radius: 'sm', tt: 'none', fw: 500 },
    }),
    Code: Code.extend({
      defaultProps: { fz: 'xs' },
    }),
  },
})

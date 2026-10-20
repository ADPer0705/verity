/**
 * Application frame: navigation, a running-jobs indicator (jobs are the
 * thing most likely to need attention mid-session) and a colour-scheme
 * switch.
 */

import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  ActionIcon,
  Anchor,
  AppShell,
  Badge,
  Burger,
  Group,
  Loader,
  NavLink,
  Text,
  ThemeIcon,
  Tooltip,
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { useComputedColorScheme, useMantineColorScheme } from '@mantine/core'
import { IconMoon, IconShieldSearch, IconSun } from '@tabler/icons-react'
import type { Icon } from '@tabler/icons-react'

export interface NavItem {
  label: string
  to: string
  icon: Icon
}

export function Shell({
  nav = [],
  activeJobs = 0,
  children,
}: {
  nav?: NavItem[]
  activeJobs?: number
  children: ReactNode
}) {
  const location = useLocation()
  const [opened, { toggle, close }] = useDisclosure(false)
  const { setColorScheme } = useMantineColorScheme()
  const scheme = useComputedColorScheme('dark')
  const hasNav = nav.length > 0
  const home = '/'

  const isActive = (to: string) => {
    if (to === '/') return location.pathname === '/' || location.pathname.startsWith('/case/')
    return location.pathname === to || location.pathname.startsWith(`${to}/`)
  }

  return (
    <AppShell
      header={{ height: 52 }}
      navbar={
        hasNav ? { width: 200, breakpoint: 'sm', collapsed: { mobile: !opened } } : undefined
      }
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            {hasNav ? (
              <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" aria-label="Menu" />
            ) : null}
            <Anchor component={Link} to={home} c="inherit" underline="never">
              <Group gap={8} wrap="nowrap">
                <ThemeIcon size="sm" variant="light" aria-hidden>
                  <IconShieldSearch size={14} />
                </ThemeIcon>
                <Text fw={600} fz="sm">
                  Verity
                </Text>
                <Text c="dimmed" fz="sm" visibleFrom="xs">
                  Deepfake examination
                </Text>
              </Group>
            </Anchor>
          </Group>

          <Group gap="xs" wrap="nowrap">
            {activeJobs > 0 ? (
              <Badge
                variant="light"
                color="indigo"
                leftSection={<Loader size={10} color="indigo" />}
                aria-label={`${activeJobs} job${activeJobs === 1 ? '' : 's'} running`}
              >
                {activeJobs} running
              </Badge>
            ) : null}
            <Tooltip label={scheme === 'dark' ? 'Light theme' : 'Dark theme'} position="bottom">
              <ActionIcon
                variant="subtle"
                color="gray"
                aria-label="Toggle colour scheme"
                onClick={() => setColorScheme(scheme === 'dark' ? 'light' : 'dark')}
              >
                {scheme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </AppShell.Header>

      {hasNav ? (
        <AppShell.Navbar p="xs">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              component={Link}
              to={item.to}
              label={item.label}
              leftSection={<item.icon size={18} stroke={1.6} />}
              active={isActive(item.to)}
              onClick={close}
              fz="sm"
            />
          ))}
        </AppShell.Navbar>
      ) : null}

      <AppShell.Main id="main-content">{children}</AppShell.Main>
    </AppShell>
  )
}

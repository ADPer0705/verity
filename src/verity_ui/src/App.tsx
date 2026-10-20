/**
 * Routing. One interface for everyone: each case opens with a
 * plain-language summary and the full examination record underneath, so
 * there is no profile to pick and nothing hidden behind one.
 */

import { Navigate, Route, Routes, useParams } from 'react-router-dom'
import { IconActivity, IconCpu, IconFolders, IconHelpCircle, IconServer } from '@tabler/icons-react'

import { api } from '@/api/client'
import { usePolling } from '@/api/hooks'
import { Shell } from '@/components/Shell'
import type { NavItem } from '@/components/Shell'

import Cases from '@/routes/Cases'
import CaseDetail from '@/routes/CaseDetail'
import Detectors from '@/routes/Detectors'
import ActivityPage from '@/routes/Activity'
import System from '@/routes/System'
import Help from '@/routes/Help'

const NAV: NavItem[] = [
  { label: 'Cases', to: '/', icon: IconFolders },
  { label: 'Detectors', to: '/detectors', icon: IconCpu },
  { label: 'Activity', to: '/activity', icon: IconActivity },
  { label: 'System', to: '/system', icon: IconServer },
  { label: 'Handling evidence', to: '/help', icon: IconHelpCircle },
]

/** Old bookmarked links from the two-profile version keep working. */
function LegacyCaseRedirect() {
  const { mediaId = '' } = useParams()
  return <Navigate to={`/case/${mediaId}`} replace />
}

export default function App() {
  // Poll only for the header's activity indicator; pages own their own data.
  const { data } = usePolling(() => api.jobs({ limit: 1 }), 6000, true, [])
  return (
    <Shell nav={NAV} activeJobs={data?.active ?? 0}>
      <Routes>
        <Route path="/" element={<Cases />} />
        <Route path="/case/:mediaId" element={<CaseDetail />} />
        <Route path="/detectors" element={<Detectors />} />
        <Route path="/activity" element={<ActivityPage />} />
        <Route path="/system" element={<System />} />
        <Route path="/help" element={<Help />} />
        <Route path="/help/:mediaId" element={<Help />} />

        <Route path="/investigator/case/:mediaId" element={<LegacyCaseRedirect />} />
        <Route path="/citizen/result/:mediaId" element={<LegacyCaseRedirect />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  )
}

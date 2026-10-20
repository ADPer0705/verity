/**
 * Translating detector output into language, for two very different readers.
 *
 * Verity's design principle is that a single real/fake number hides the
 * disagreement that matters. Both profiles therefore get the same underlying
 * consensus object; what changes is the vocabulary and how much of the
 * machinery is on screen. Neither wording ever claims certainty the panel of
 * detectors did not produce, and the plain-language copy always ends somewhere
 * actionable rather than at a verdict.
 */

import type { Classification, Consensus, ConsensusBand } from '@/api/types'

export type ToneKind = 'critical' | 'caution' | 'neutral' | 'reassuring' | 'unknown'

export interface BandPresentation {
  /** Short label for a tag or table cell. */
  label: string
  /** Citizen-facing headline — plain language, no jargon. */
  plainHeadline: string
  /** Citizen-facing explanation of what the result does and does not mean. */
  plainBody: string
  /** Investigator-facing one-liner, stated as an evidential position. */
  technicalSummary: string
  tone: ToneKind
  /** Badge colour (a key of the Mantine palette). */
  tagType: 'red' | 'pink' | 'violet' | 'cyan' | 'teal' | 'green' | 'gray'
}

export const BAND_PRESENTATION: Record<ConsensusBand, BandPresentation> = {
  strong_manipulation: {
    label: 'Signs of manipulation',
    plainHeadline: 'This file shows strong signs of being manipulated',
    plainBody:
      'Most of the checks that ran agreed that parts of this file look artificially generated or altered. That is a serious signal, but it is not a legal finding — an investigator still needs to confirm it.',
    technicalSummary:
      'Majority of eligible detectors classified the media as manipulated, with a decisive mean probability.',
    tone: 'critical',
    tagType: 'red',
  },
  likely_manipulation: {
    label: 'Leaning manipulated',
    plainHeadline: 'This file may have been manipulated',
    plainBody:
      'More checks pointed towards manipulation than towards authenticity, but the result is not clear-cut. Treat this as a reason to preserve the file and report it, not as proof.',
    technicalSummary:
      'More detectors classified the media as manipulated than authentic; margin is not decisive.',
    tone: 'caution',
    tagType: 'pink',
  },
  inconclusive: {
    label: 'Inconclusive',
    plainHeadline: 'The checks disagreed about this file',
    plainBody:
      'Different analysis methods reached different conclusions. That happens with heavily compressed, re-shared or low-resolution media. It does not mean the file is safe, and it does not mean it is fake — it means an expert needs to look at it.',
    technicalSummary:
      'Detectors split materially. Do not report a single conclusion; cite the per-detector findings and the media conditions that explain the divergence.',
    tone: 'neutral',
    tagType: 'violet',
  },
  likely_authentic: {
    label: 'Leaning authentic',
    plainHeadline: 'No clear signs of manipulation were found',
    plainBody:
      'Most checks did not find evidence that this file was artificially generated or altered. Tools cannot prove a file is genuine — they can only say they found nothing suspicious in what they examined.',
    technicalSummary:
      'More detectors classified the media as authentic than manipulated; absence of detection is not evidence of authenticity.',
    tone: 'reassuring',
    tagType: 'teal',
  },
  strong_authentic: {
    label: 'No manipulation detected',
    plainHeadline: 'The checks found no signs of manipulation',
    plainBody:
      'The methods that ran consistently found nothing that looks artificially generated. This is not a guarantee of authenticity: a manipulation technique none of these methods was built to catch would not show up here.',
    technicalSummary:
      'Consistent authentic classification across eligible detectors. Coverage is bounded by the detectors that were eligible — see the exclusion reasons.',
    tone: 'reassuring',
    tagType: 'green',
  },
  no_analysis: {
    label: 'Not examined yet',
    plainHeadline: 'This file has not been checked yet',
    plainBody: 'Start the check to see what the analysis finds.',
    technicalSummary: 'No findings compiled for this media item.',
    tone: 'unknown',
    tagType: 'gray',
  },
}

export function presentBand(band: ConsensusBand): BandPresentation {
  return BAND_PRESENTATION[band] ?? BAND_PRESENTATION.no_analysis
}

/**
 * How much weight the interface should let the reader put on the result.
 *
 * Confidence in the *aggregate* is a function of agreement and coverage, not
 * of the individual detectors' self-reported confidence — a unanimous panel
 * of two says less than a near-unanimous panel of eight.
 */
export function reliability(consensus: Consensus): {
  level: 'high' | 'moderate' | 'low' | 'none'
  label: string
  explanation: string
} {
  if (consensus.total === 0) {
    return { level: 'none', label: 'No analysis', explanation: 'No detector has produced a finding.' }
  }
  if (consensus.disagreement && consensus.agreement < 0.75) {
    return {
      level: 'low',
      label: 'Low — methods disagree',
      explanation: `${consensus.fake} of ${consensus.total} methods reported manipulation and ${consensus.real} reported authentic. A split panel cannot support a single conclusion.`,
    }
  }
  if (consensus.total < 3) {
    return {
      level: 'low',
      label: 'Low — narrow coverage',
      explanation: `Only ${consensus.total} method${consensus.total === 1 ? '' : 's'} could run on this media, so the result rests on a narrow base.`,
    }
  }
  if (consensus.agreement >= 0.85) {
    return {
      level: 'high',
      label: 'High — methods agree',
      explanation: `${Math.round(consensus.agreement * 100)}% of the ${consensus.total} methods that ran reached the same conclusion.`,
    }
  }
  return {
    level: 'moderate',
    label: 'Moderate',
    explanation: `${Math.round(consensus.agreement * 100)}% of the ${consensus.total} methods that ran reached the same conclusion.`,
  }
}

export const CLASSIFICATION_TAG: Record<Classification, 'red' | 'green' | 'gray'> = {
  FAKE: 'red',
  REAL: 'green',
  UNCERTAIN: 'gray',
}

/** Plain-language label for a single detector's classification. */
export const CLASSIFICATION_PLAIN: Record<Classification, string> = {
  FAKE: 'Signs of manipulation',
  REAL: 'No signs of manipulation',
  UNCERTAIN: 'Could not decide',
}

/**
 * Limitations that always apply, stated up front rather than buried.
 * The README commits to documenting limitations; this is where the UI keeps
 * that promise on every result screen.
 */
export function standingLimitations(consensus: Consensus, mediaType: string): string[] {
  const limitations = [
    'These are automated signals, not a legal determination. A qualified examiner must review the findings before they are relied on.',
    'Detectors only recognise manipulation techniques they were trained on. A newer technique can pass unnoticed.',
  ]
  if (consensus.disagreement) {
    limitations.push(
      'The methods that ran did not agree with each other, so no single conclusion is supported by this examination alone.',
    )
  }
  if (consensus.total > 0 && consensus.total < 3) {
    limitations.push(
      'Few methods were eligible for this file, which narrows how much the result covers.',
    )
  }
  if (mediaType === 'video') {
    limitations.push(
      'Video is examined by sampling frames and the faces within them. Manipulation confined to unsampled frames can be missed.',
    )
  }
  if (mediaType === 'audio') {
    limitations.push('Audio-only examination is limited to the detectors that accept audio input.')
  }
  return limitations
}

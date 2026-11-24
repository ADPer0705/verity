import { defineConfig } from 'vitepress'

export default defineConfig({
  srcDir: 'docs',
  cleanUrls: true,
  lastUpdated: false,
  appearance: 'dark',
  title: 'Verity',
  titleTemplate: ':title — Verity Wiki',
  description:
    'Multi-detector deepfake examination for single-investigator workstations and air-gapped forensic labs.',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#4c6ef5' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'Verity Documentation' }],
    ['meta', { property: 'og:title', content: 'Verity — Forensic Deepfake Examination Workbench' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'Standalone air-gap binary, 58 isolated detector architectures, model-free forensic analyzers, and SHA-256 chain-of-custody bundles.',
      },
    ],
    ['meta', { property: 'og:url', content: 'https://verity.dearanayji.rocks/' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    [
      'link',
      {
        rel: 'stylesheet',
        href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap',
      },
    ],
  ],

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Verity',

    search: {
      provider: 'local',
      options: {
        detailedView: true,
      },
    },

    nav: [
      { text: 'Quickstart', link: '/quickstart', activeMatch: '/quickstart' },
      { text: 'Architecture', link: '/architecture/overview', activeMatch: '/architecture/' },
      { text: '58 Detectors', link: '/reference/detector-catalog', activeMatch: '/reference/' },
      { text: 'Air-Gap Ops', link: '/operations/deployment-and-airgap', activeMatch: '/operations/' },
      {
        text: 'v0.6.0',
        items: [
          {
            text: 'GitHub Release (v0.6.0)',
            link: 'https://github.com/ADPer0705/verity/releases/tag/v0.6.0',
          },
          {
            text: 'Hugging Face Weights',
            link: 'https://huggingface.co/ADPer/verity-weights',
          },
          {
            text: 'Container (GHCR)',
            link: 'https://github.com/ADPer0705/verity/pkgs/container/verity',
          },
        ],
      },
    ],

    sidebar: [
      {
        text: 'Start Here',
        items: [
          { text: 'Quickstart & Map', link: '/quickstart' },
          { text: 'Architecture Overview', link: '/architecture/overview' },
          { text: 'Examination Pipeline', link: '/architecture/pipeline' },
        ],
      },
      {
        text: 'Core Architecture',
        collapsed: false,
        items: [
          { text: 'Ingest & Media Analysis', link: '/architecture/ingest-and-analysis' },
          { text: 'Artifact Extraction', link: '/architecture/artifact-extraction' },
          { text: 'Model-Free Forensics', link: '/architecture/forensics' },
          { text: 'Detector Selection & Tiers', link: '/architecture/detector-selection' },
          { text: 'Detector Runtime & uv Isolation', link: '/architecture/detector-runtime' },
          { text: 'Shared Detector Engine', link: '/architecture/detector-engine' },
          { text: 'Findings & Consensus', link: '/architecture/findings-and-consensus' },
          { text: 'Report Generation', link: '/architecture/reporting' },
          { text: 'Security & Chain of Custody', link: '/architecture/security-and-integrity' },
          { text: 'Web API & Console', link: '/architecture/web-api-and-console' },
          { text: 'Weights & Checkpoints', link: '/architecture/weights-and-checkpoints' },
        ],
      },
      {
        text: 'Reference',
        collapsed: false,
        items: [
          { text: 'All Reference Pages', link: '/reference/' },
          { text: 'Detector Catalog (58 Ports)', link: '/reference/detector-catalog' },
          { text: 'Explainability & Formats', link: '/reference/explainability-artifacts' },
        ],
      },
      {
        text: 'Operations & Releases',
        collapsed: false,
        items: [
          { text: 'All Operations Pages', link: '/operations/' },
          { text: 'Deployment & Air-Gap Guide', link: '/operations/deployment-and-airgap' },
          { text: 'Releasing & Extending Verity', link: '/operations/releasing-and-publishing' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: 'On this page',
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/ADPer0705/verity' }],

    docFooter: {
      prev: 'Previous page',
      next: 'Next page',
    },

    footer: {
      message: 'Verity v0.6.0 · Non-commercial forensic workbench (CC BY-NC 4.0) · Grounded in OpenWiki',
      copyright: 'All detector weights remain the intellectual property of their respective upstream authors.',
    },
  },
})

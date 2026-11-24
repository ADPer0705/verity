#!/usr/bin/env node
/**
 * Stage OpenWiki Markdown pages from ../openwiki into gitignored website/docs/
 * so VitePress builds directly from openwiki/ with zero duplicated prose.
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const WEBSITE_DIR = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(WEBSITE_DIR, '..')
const OPENWIKI_DIR = path.join(REPO_ROOT, 'openwiki')
const DOCS_DIR = path.join(WEBSITE_DIR, 'docs')
const PUBLIC_SRC = path.join(WEBSITE_DIR, 'public')
const PUBLIC_DST = path.join(DOCS_DIR, 'public')
const HOME_SRC = path.join(WEBSITE_DIR, 'index.md')

const SECTION_INDEX_META = {
  'architecture/index.md': {
    title: 'Architecture',
    description: 'Pipeline stages, package boundaries, model-free forensics, detector isolation, and integrity invariants.',
    heading: 'Core Architecture & Pipeline',
    lead: 'Engineering design rationale, unidirectional boundaries, isolated detector execution, and evidence preservation across Verity.',
  },
  'reference/index.md': {
    title: 'Reference',
    description: 'Catalog of all 58 ported detectors, trust tiers, explainability visuals, and on-disk artifact formats.',
    heading: 'Reference & Catalogs',
    lead: 'Complete inventory of all 58 ported detector architectures alongside every JSON record, visual plot, and export bundle format.',
  },
  'operations/index.md': {
    title: 'Operations',
    description: 'Standalone binary installation, air-gapped provisioning, USB transfer kits, configuration, and release runbooks.',
    heading: 'Deployment, Air-Gap & Release Operations',
    lead: 'Field-ready workstation deployment guides, offline air-gap workflows, and maintainer release engineering runbooks.',
  },
}

function escapeNonCodeTags(markdown) {
  // Preserve YAML frontmatter intact, and escape rare raw <placeholder> tags
  // outside fenced code blocks and inline backticks so VitePress Vue compiler
  // does not interpret them as custom Vue elements.
  let frontmatter = ''
  let body = markdown
  if (markdown.startsWith('---\n')) {
    const endIdx = markdown.indexOf('\n---\n', 4)
    if (endIdx !== -1) {
      frontmatter = markdown.slice(0, endIdx + 5)
      body = markdown.slice(endIdx + 5)
    }
  }

  const lines = body.split('\n')
  let inFence = false
  let fenceChar = ''
  const out = []

  for (const line of lines) {
    const trimmed = line.trimStart()
    const fenceMatch = trimmed.match(/^(`{3,}|~{3,})/)
    if (fenceMatch) {
      const marker = fenceMatch[1][0]
      if (!inFence) {
        inFence = true
        fenceChar = marker
      } else if (marker === fenceChar) {
        inFence = false
        fenceChar = ''
      }
      out.push(line)
      continue
    }

    if (inFence) {
      out.push(line)
      continue
    }

    // Split by inline code spans so backticked snippets remain untouched
    const parts = line.split(/(`+[^`]*`+)/g)
    const processed = parts
      .map((part, idx) => {
        if (idx % 2 === 1) return part
        // Escape uppercase/hyphenated/numeric placeholder tags like <12-char-id>, <id>, <dir>
        // while keeping standard HTML comments & common tags if intentionally placed.
        return part.replace(/<(?!\/?(?:div|span|p|a|br|hr|em|strong|code|pre|ul|ol|li|table|thead|tbody|tr|th|td|img|sup|sub|details|summary|!--)\b)([A-Za-z0-9_.-][^>\n]{0,40})>/g, '&lt;$1&gt;')
      })
      .join('')
    out.push(processed)
  }

  return frontmatter + out.join('\n')
}

function enhanceSectionIndex(relPosix, content) {
  const meta = SECTION_INDEX_META[relPosix]
  if (!meta) return content
  const body = content.replace(/^#\s+Files\s*\n+/m, '')
  return [
    '---',
    `title: ${meta.title}`,
    `description: ${meta.description}`,
    '---',
    '',
    `# ${meta.heading}`,
    '',
    meta.lead,
    '',
    '## Pages in this section',
    '',
    body.trimStart(),
    '',
  ].join('\n')
}

function copyDirRecursive(srcDir, dstDir) {
  fs.mkdirSync(dstDir, { recursive: true })
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const srcPath = path.join(srcDir, entry.name)
    const dstPath = path.join(dstDir, entry.name)
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, dstPath)
    } else if (entry.isFile()) {
      fs.copyFileSync(srcPath, dstPath)
    }
  }
}

function stageOpenWiki(srcRoot, dstRoot) {
  let count = 0
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.')) continue
      const absPath = path.join(dir, entry.name)
      const rel = path.relative(srcRoot, absPath)
      const relPosix = rel.split(path.sep).join('/')
      if (entry.isDirectory()) {
        walk(absPath)
        continue
      }
      if (!entry.isFile() || !entry.name.endsWith('.md')) continue
      // Root openwiki/index.md is replaced by custom VitePress landing page website/index.md
      if (relPosix === 'index.md') continue

      const outPath = path.join(dstRoot, rel)
      fs.mkdirSync(path.dirname(outPath), { recursive: true })
      let content = fs.readFileSync(absPath, 'utf8')
      if (SECTION_INDEX_META[relPosix]) {
        content = enhanceSectionIndex(relPosix, content)
      }
      content = escapeNonCodeTags(content)
      fs.writeFileSync(outPath, content, 'utf8')
      count += 1
    }
  }
  walk(srcRoot)
  return count
}

if (!fs.existsSync(OPENWIKI_DIR)) {
  throw new Error(`OpenWiki directory not found: ${OPENWIKI_DIR}`)
}

fs.rmSync(DOCS_DIR, { recursive: true, force: true })
fs.mkdirSync(DOCS_DIR, { recursive: true })

const stagedCount = stageOpenWiki(OPENWIKI_DIR, DOCS_DIR)
fs.copyFileSync(HOME_SRC, path.join(DOCS_DIR, 'index.md'))

if (fs.existsSync(PUBLIC_SRC)) {
  copyDirRecursive(PUBLIC_SRC, PUBLIC_DST)
}

console.log(`[stage_wiki] staged ${stagedCount} OpenWiki pages + landing page -> ${path.relative(REPO_ROOT, DOCS_DIR)}`)

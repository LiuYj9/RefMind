<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import katex from 'katex'
const props = defineProps<{ text: string }>()
const emit = defineEmits<{ citation: [href: string] }>()
const html = computed(() => {
  const formulas: string[] = []
  const source = props.text.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\$([^\n$]+?)\$|\\\(([^\n]+?)\\\)/g, (_all, block1, block2, inline1, inline2) => {
    try { const value = katex.renderToString(block1 || block2 || inline1 || inline2, { displayMode: !!(block1 || block2), throwOnError: false, trust: false, output: 'html' }); formulas.push(value); return `REFMINDFORMULA${formulas.length - 1}END` } catch { return _all }
  })
  let result = DOMPurify.sanitize(marked.parse(source, { async: false, breaks: true }) as string, {
    FORBID_TAGS: ['img', 'svg', 'math', 'iframe', 'object', 'embed', 'video', 'audio', 'source', 'link', 'form', 'style', 'script'],
    FORBID_ATTR: ['style', 'src', 'srcset', 'srcdoc'],
  })
  result = result.replace(/REFMINDFORMULA(\d+)END/g, (_m, index) => formulas[Number(index)] || '')
  // External image URLs never trigger browser requests from generated content.
  return DOMPurify.sanitize(result, { FORBID_TAGS: ['img', 'iframe', 'form', 'style', 'script'], FORBID_ATTR: ['srcdoc', 'onload', 'onclick'] })
})
function clicked(event: MouseEvent) {
  const link = (event.target as HTMLElement).closest('a')
  if (!link) return
  const href = link.getAttribute('href') || ''
  if (href.includes('open_citation=') || href.includes('open_resource=')) { event.preventDefault(); emit('citation', href) }
  else { link.setAttribute('target', '_blank'); link.setAttribute('rel', 'noopener noreferrer') }
}
</script>
<template><div class="answer-markdown" @click="clicked" v-html="html"></div></template>

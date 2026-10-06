// Toast, copy, code example tabs, citation, and dialog behavior shared by pages.
import { $, $$, escapeHTML } from './dom.js';

let toastTimer;
export function toast(message) {
  $('#toast').textContent = message; $('#toast').classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 3000);
}
const examples = {
  python: `from sol_metric import SOL, TextEncoder\n\nencoder = TextEncoder('gpt2-large')\nmetric = SOL(device='auto')\n\n# Two nonempty lists of text samples\nscore = metric.from_texts(\n    reference_texts, generated_texts,\n    encoder=encoder\n)\nprint(score)  # Lower means closer.`,
  install: `# With GPU support and text encoders\npip install "sol-metric[torch,text]"\n\n# NumPy only, for existing embeddings\npip install sol-metric`,
};
let currentExample = 'python';
function setCode(key) {
  currentExample = key;
  $('#code-example').innerHTML = examples[key].split('\n').map(line => {
    if (line.startsWith('#')) return `<span class="code-comment">${escapeHTML(line)}</span>`;
    return escapeHTML(line).replace(/(&#39;.*?&#39;)/g, '<span class="code-string">$1</span>').replace(/\b(from|import)\b/g, '<span class="code-keyword">$1</span>');
  }).join('\n');
  $$('[data-code]').forEach(button => { button.setAttribute('aria-selected', String(button.dataset.code === key)); button.tabIndex = button.dataset.code === key ? 0 : -1; });
  $('#code-content').setAttribute('aria-labelledby', `code-${key}`);
}
export async function copy(text, message) {
  try { await navigator.clipboard.writeText(text); toast(message); }
  catch { toast('Copy unavailable. Select the text to copy it.'); }
}

export function initSiteChrome() {
  // Pages without a Get started section have no code example.
  if ($('#code-example')) {
    setCode('python');
    $$('[data-code]').forEach(button => button.addEventListener('click', () => setCode(button.dataset.code)));
    $('.code-toolbar [role=tablist]').addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      setCode(event.key === 'Home' ? 'python' : event.key === 'End' ? 'install' : currentExample === 'python' ? 'install' : 'python');
      $(`#code-${currentExample}`).focus();
    });
    $('#copy-code').addEventListener('click', () => copy(examples[currentExample], 'Code copied.'));
  }
  $$('[data-cite]').forEach(button => button.addEventListener('click', () => $('#citation-dialog').showModal()));
  $('#copy-citation').addEventListener('click', () => copy($('#citation-text').textContent, 'BibTeX copied'));
  $$('.dialog-close').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
  $$('dialog').forEach(dialog => dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  }));
}

// Dependency-free SVG charts + a small markdown renderer.
const Charts = (() => {
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

  // opts: { labels:[], series:[{name,color,values:[]}], thresholds:[{value,color,label}],
  //         yLabel, logScale, height }
  function line(opts) {
    const W = 720, H = opts.height || 260;
    const m = { t: 16, r: 16, b: 46, l: 56 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const labels = opts.labels || [];
    const series = opts.series || [];
    const thresholds = opts.thresholds || [];

    // Collect all values (series + thresholds) for the y-domain.
    let vals = [];
    series.forEach(s => s.values.forEach(v => { if (v != null) vals.push(v); }));
    thresholds.forEach(t => vals.push(t.value));
    if (!vals.length) return '<div class="muted">No data to plot yet.</div>';

    const log = !!opts.logScale;
    let min = Math.min(...vals), max = Math.max(...vals);
    if (log) { min = Math.max(min, 0.1); }
    if (min === max) { max = min + 1; }
    const pad = (max - min) * 0.08;
    let lo = log ? min * 0.7 : min - pad;
    let hi = log ? max * 1.3 : max + pad;
    if (!log) lo = Math.min(lo, 0);

    const tY = (v) => {
      if (v == null) return null;
      const f = log
        ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
        : (v - lo) / (hi - lo);
      return m.t + ih - f * ih;
    };
    const tX = (i) => labels.length <= 1 ? m.l + iw / 2 : m.l + (i / (labels.length - 1)) * iw;

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    // gridlines + y ticks
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const v = log
        ? Math.pow(10, Math.log10(lo) + (i / ticks) * (Math.log10(hi) - Math.log10(lo)))
        : lo + (i / ticks) * (hi - lo);
      const y = tY(v);
      svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
      svg += `<text x="${m.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${fmtTick(v)}</text>`;
    }
    // threshold lines
    thresholds.forEach(t => {
      const y = tY(t.value);
      svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${t.color}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
      svg += `<text x="${W - m.r}" y="${y - 4}" text-anchor="end" font-size="10" fill="${t.color}">${esc(t.label)}</text>`;
    });
    // x labels
    labels.forEach((lb, i) => {
      const x = tX(i);
      svg += `<text x="${x}" y="${H - 22}" text-anchor="middle" font-size="10" fill="var(--muted)">${esc(lb)}</text>`;
    });
    // series
    series.forEach(s => {
      let d = '', started = false;
      s.values.forEach((v, i) => {
        const y = tY(v);
        if (y == null) { started = false; return; }
        d += (started ? ' L' : ' M') + tX(i).toFixed(1) + ' ' + y.toFixed(1);
        started = true;
      });
      svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.5"/>`;
      s.values.forEach((v, i) => {
        const y = tY(v);
        if (y == null) return;
        svg += `<circle cx="${tX(i)}" cy="${y}" r="4" fill="${s.color}"><title>${esc(s.name)}: ${v}</title></circle>`;
      });
    });
    if (opts.yLabel) {
      svg += `<text x="14" y="${m.t + ih / 2}" transform="rotate(-90 14 ${m.t + ih / 2})" text-anchor="middle" font-size="11" fill="var(--muted)">${esc(opts.yLabel)}</text>`;
    }
    svg += '</svg>';
    return svg;
  }

  function fmtTick(v) {
    if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 'k';
    if (v >= 10) return v.toFixed(0);
    return v.toFixed(1);
  }

  // ---- Tiny markdown renderer (headings, lists, code, links, bold/italic, tables, hr) ----
  function markdown(src) {
    const lines = src.replace(/\r\n/g, '\n').split('\n');
    let html = '', i = 0;
    const inline = (t) => esc(t)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    while (i < lines.length) {
      const line = lines[i];
      if (/^```/.test(line)) {
        let code = ''; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { code += lines[i] + '\n'; i++; }
        i++;
        html += `<pre><code>${esc(code)}</code></pre>`;
        continue;
      }
      if (/^#{1,6}\s/.test(line)) {
        const lvl = line.match(/^#+/)[0].length;
        html += `<h${lvl}>${inline(line.replace(/^#+\s/, ''))}</h${lvl}>`;
        i++; continue;
      }
      if (/^\s*[-*]\s/.test(line)) {
        html += '<ul>';
        while (i < lines.length && /^\s*[-*]\s/.test(lines[i])) {
          html += `<li>${inline(lines[i].replace(/^\s*[-*]\s/, ''))}</li>`; i++;
        }
        html += '</ul>'; continue;
      }
      if (/^\s*\d+\.\s/.test(line)) {
        html += '<ol>';
        while (i < lines.length && /^\s*\d+\.\s/.test(lines[i])) {
          html += `<li>${inline(lines[i].replace(/^\s*\d+\.\s/, ''))}</li>`; i++;
        }
        html += '</ol>'; continue;
      }
      if (/^>\s?/.test(line)) {
        html += `<blockquote>${inline(line.replace(/^>\s?/, ''))}</blockquote>`; i++; continue;
      }
      if (/^(\s*[-*_]){3,}\s*$/.test(line)) { html += '<hr>'; i++; continue; }
      if (/^\|.*\|/.test(line) && i + 1 < lines.length && /^\|[-:| ]+\|/.test(lines[i + 1])) {
        const head = line.split('|').slice(1, -1).map(s => s.trim());
        i += 2;
        let rows = '';
        while (i < lines.length && /^\|.*\|/.test(lines[i])) {
          const cells = lines[i].split('|').slice(1, -1).map(s => inline(s.trim()));
          rows += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>'; i++;
        }
        html += '<table><thead><tr>' + head.map(h => `<th>${inline(h)}</th>`).join('') + '</tr></thead><tbody>' + rows + '</tbody></table>';
        continue;
      }
      if (line.trim() === '') { i++; continue; }
      // paragraph (accumulate until blank)
      let para = line; i++;
      while (i < lines.length && lines[i].trim() !== '' && !/^(#{1,6}\s|```|>\s?|\s*[-*]\s|\s*\d+\.\s|\|)/.test(lines[i])) {
        para += ' ' + lines[i]; i++;
      }
      html += `<p>${inline(para)}</p>`;
    }
    return html;
  }

  return { line, markdown };
})();

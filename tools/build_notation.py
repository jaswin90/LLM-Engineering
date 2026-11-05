"""Regenerate docs/notation.html from NOTATION.md (run from the repo root)."""
import re
s = open('NOTATION.md').read()
out, intable = [], False
def cell(c):
    c = c.strip().replace('<', '&lt;')
    c = re.sub(r'`([^`]+)`', r'<code>\1</code>', c)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', c)
for line in s.split('\n'):
    if line.startswith('|'):
        if set(line.replace('|', '').strip()) <= set('-: '):
            continue
        cells = line.strip().strip('|').split('|')
        if not intable:
            out.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th>%s</th>' % cell(c) for c in cells) + '</tr></thead><tbody>')
            intable = True
        else:
            out.append('<tr>' + ''.join('<td>%s</td>' % cell(c) for c in cells) + '</tr>')
        continue
    if intable:
        out.append('</tbody></table></div>'); intable = False
    if line.startswith('# '):
        continue
    if line.startswith('## '):
        out.append('<h2>%s</h2>' % line[3:])
    elif line.startswith('- '):
        out.append('<p>• %s</p>' % cell(line[2:]))
    elif line.strip():
        out.append('<p>%s</p>' % cell(line))
if intable:
    out.append('</tbody></table></div>')
head = open('docs/notation.html').read().split('<div class="cover">')[0]
page = head + '<div class="cover">\n<h1>Notation</h1>\n<p class="sub">One set of symbols is used throughout the book. Every symbol a chapter introduces is listed here, and no symbol has two meanings.</p>\n' + '\n'.join(out) + '\n</div>\n</body>\n</html>\n'
open('docs/notation.html', 'w').write(page)
print('wrote docs/notation.html')

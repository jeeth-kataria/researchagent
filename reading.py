"""Deterministic section outlines and bounded, auditable source delivery."""
import copy
import re

HEADING = re.compile(r'^(?:\d+(?:\.\d+)*[.)]?\s+)?(abstract|introduction|background|related work|methods?|methodology|approach|experimental setup|experiments?|evaluation|results(?: and discussion)?|discussion|limitations?|conclusions?|references|appendix)(?:\s*[:.]\s*)?$', re.I)


def sections_for(pages, evidence):
    markers = []
    for page, text in pages.items():
        if re.search(r'^\s*(table of )?contents\s*$', text, re.I | re.M):
            continue  # A contents listing is not section body text.
        offset = 0
        for line in text.splitlines(keepends=True):
            match = HEADING.fullmatch(line.strip())
            if match:
                markers.append((page, offset, line.strip(), match.group(1).lower()))
            offset += len(line)
    reliable = len({item[3] for item in markers}) >= 2
    if not reliable:
        markers = [(page, 0, f'Physical page {page}', 'page') for page, text in pages.items() if text.strip()]
    elif markers and (markers[0][0], markers[0][1]) != (min(pages), 0):
        markers.insert(0, (min(pages), 0, 'Opening text', 'opening'))
    sections = []
    for index, (page, start, title, kind) in enumerate(markers):
        finish = markers[index + 1][:2] if index + 1 < len(markers) else (max(pages) + 1, 0)
        ids = [eid for eid, entry in evidence.items()
               if (entry['page'], entry['end']) > (page, start)
               and (entry['page'], entry['start']) < finish]
        if not ids:
            continue
        # The preview is a separate, explicitly partial source excerpt.
        preview = pages[page][start:start + 80]
        sections.append({'section_id': f'S{len(sections)+1:03d}', 'title': title,
            'kind': kind, 'page': page, 'start': start, 'end_page': finish[0],
            'end': finish[1], 'preview': preview, 'evidence_ids': ids})
    return {'mode': 'heading_heuristics' if reliable else 'physical_pages', 'sections': sections}


def reading_packet(role, state, maximum):
    sections = state['reading']['sections']
    # Start at evenly spaced sections so a small budget is not spent only at the beginning.
    slots = min(len(sections), max(1, (maximum - 800) // 800))
    spaced = list(dict.fromkeys(round(i * (len(sections)-1) / max(1, slots-1)) for i in range(slots))) if sections else []
    order = spaced + [i for i in range(len(sections)) if i not in spaced]
    if role == 'analyst':
        focus = [i for i, section in enumerate(sections) if re.search(r'method|approach|experiment|evaluation|result', section['kind'])]
        order = focus + [i for i in order if i not in focus]
    candidates = []
    if role == 'reviewer':
        for field in ('report_draft', 'reader_findings', 'analyst_findings'):
            for finding in state[field].get('findings', []):
                candidates.extend(finding['evidence_ids'])
        previously = {eid for record in state['coverage'].values() for eid in record['evidence_ids']}
        candidates.extend(eid for eid in state['evidence'] if eid not in previously)
    # Round-robin across regions, not all of one section before the next.
    queues = [sections[i]['evidence_ids'] for i in order]
    for depth in range(max(map(len, queues), default=0)):
        candidates.extend(queue[depth] for queue in queues if depth < len(queue))
    selected, used = [], 0
    for eid in dict.fromkeys(candidates):
        entry = state['evidence'][eid]
        if role != 'writer' and used + len(entry['text']) <= maximum - 800:
            selected.append(copy.deepcopy(entry))
            used += len(entry['text'])
    outline = [{key: value for key, value in section.items() if key in
                ('section_id', 'title', 'kind', 'page', 'preview', 'evidence_ids')}
               for section in sections[:24]]
    return {'outline_mode': state['reading']['mode'], 'outline': outline,
            'outline_sections_omitted': max(0, len(sections)-24), 'source_passages': selected,
            'source_character_limit': maximum,
            'note': 'Previews are partial excerpts. Only source_passages contain full registry passages. Coverage is delivery, not understanding.'}


def bound_tool_result(name, result, remaining):
    entries = result['matches'] if name == 'search_paper' else [result]
    selected, used = [], 0
    for entry in entries:
        if used + len(entry['text']) <= remaining:
            selected.append(entry)
            used += len(entry['text'])
    if name == 'search_paper':
        return {**result, 'matches': selected, 'omitted_by_context_limit': len(entries)-len(selected)}, selected
    if not selected:
        return {'evidence_id': result['evidence_id'], 'unavailable': 'Source context limit reached; no passage supplied.'}, []
    return result, selected


def record_delivery(state, role, entries, preview_ids=()):
    record = state['coverage'][role]
    record['source_characters'] += sum(len(entry['text']) for entry in entries)
    record['evidence_ids'] = list(dict.fromkeys(record['evidence_ids'] + [e['evidence_id'] for e in entries]))
    record['preview_section_ids'] = list(dict.fromkeys(record['preview_section_ids'] + list(preview_ids)))


def span_characters(spans):
    total = 0
    for page in {span[0] for span in spans}:
        end = 0
        for _, start, stop in sorted(span for span in spans if span[0] == page):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
    return total


def coverage_summary(state):
    total = sum(len(text) for text in state['pages'].values())
    roles = {}
    for role, record in state['coverage'].items():
        received = set(record['evidence_ids'])
        full = [(state['evidence'][eid]['page'], state['evidence'][eid]['start'],
                 state['evidence'][eid]['end']) for eid in received]
        previews = [(section['page'], section['start'], section['start']+len(section['preview']))
                    for section in state['reading']['sections']
                    if section['section_id'] in record['preview_section_ids']]
        unique = span_characters(full)
        any_form = span_characters(full + previews)
        roles[role] = {**record, 'unique_source_characters': unique,
                       'preview_characters': span_characters(previews),
                       'characters_supplied_in_any_form': any_form,
                       'characters_not_supplied': total-any_form,
                       'full_passage_characters_not_supplied': total-unique,
                       'evidence_not_supplied': [eid for eid in state['evidence'] if eid not in received]}
    return {'extracted_characters': total, 'passages': len(state['evidence']),
            'outline_mode': state['reading']['mode'], 'sections': len(state['reading']['sections']),
            'roles': roles, 'note': 'Source spans are deduplicated; previews remain distinct from full passages. Successful response confirms submission, not comprehension or a full review.'}

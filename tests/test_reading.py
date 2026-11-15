import json
from unittest.mock import Mock
import runpy
from pathlib import Path

import pytest

from agents import Limits
from reading import sections_for, reading_packet, coverage_summary
from workflow import initial_state, build_graph

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_workflow.py')))


def document():
    return initial_state('What supports the claim?', {
        1: 'Abstract\n' + 'opening text. ' * 65,
        2: '1 Introduction\n' + 'research problem. ' * 60,
        3: '2 Methods\n' + 'dataset baseline metric. ' * 55,
        4: '3 Results\n' + 'measured values. ' * 70,
        5: '4 Limitations\n' + 'one machine. ' * 90,
        6: '5 Conclusion\n' + 'conclusions. ' * 70})


def test_headings_and_exact_passages_across_sections():
    state = document()
    assert state['reading']['mode'] == 'heading_heuristics'
    assert [section['kind'] for section in state['reading']['sections']] == ['abstract', 'introduction', 'methods', 'results', 'limitations', 'conclusion']
    for entry in state['evidence'].values():
        assert entry['text'] == state['pages'][entry['page']][entry['start']:entry['end']]
    packet = reading_packet('reader', state, 4000)
    assert len({entry['page'] for entry in packet['source_passages']}) >= 4
    assert 6 in {entry['page'] for entry in packet['source_passages']}
    assert sum(len(entry['text']) for entry in packet['source_passages']) <= 3200


@pytest.mark.parametrize('pages', [{1: 'No known headings.\nOnly prose.', 2: 'Another page.'},
    {1: 'Introduction\nOne heading does not establish reliable sections.'},
    {1: 'Contents\nIntroduction\nMethods\nResults', 2: 'Just body text without headings.'}])
def test_unreliable_headings_use_physical_pages(pages):
    state = initial_state('Explain.', pages)
    assert state['reading']['mode'] == 'physical_pages'
    assert len(state['reading']['sections']) == len(pages)


def test_analyst_focuses_on_experiments_and_reviewer_revisits_draft():
    state = document()
    packet = reading_packet('analyst', state, 2400)
    assert packet['source_passages'][0]['page'] == 3
    last = next(eid for eid, entry in state['evidence'].items() if entry['page'] == 6)
    state['report_draft'] = {'findings': [{'evidence_ids': [last]}]}
    assert reading_packet('reviewer', state, 2400)['source_passages'][0]['evidence_id'] == last
    assert reading_packet('writer', state, 2400)['source_passages'] == []


def test_previews_alone_never_count_as_full_passage_coverage():
    state = document()
    packet = reading_packet('reader', state, 1600)
    assert packet['outline']
    assert coverage_summary(state)['roles']['reader']['unique_source_characters'] == 0
    assert coverage_summary(state)['roles']['reader']['characters_not_supplied'] == sum(map(len,state['pages'].values()))


def test_outline_and_source_budget_omit_explicitly():
    state = initial_state('Read.', {page: 'x' * 801 for page in range(1, 31)})
    packet = reading_packet('reader', state, 1600)
    assert len(packet['outline']) == 24 and packet['outline_sections_omitted'] == 6
    assert sum(len(entry['text']) for entry in packet['source_passages']) <= 800


def test_real_graph_records_only_successful_input_delivery_and_source_limits():
    helpers = HELPERS
    client = Mock(api_key='mock-only')
    outputs = helpers['complete_run']()
    outputs[0] = helpers['response'](calls=[helpers['tool']('get_evidence', {'evidence_id':'E0001'})])
    client.chat.completions.create.side_effect = outputs
    state = document()
    result = build_graph(client, 'openai/gpt-oss-20b', Limits(source_chars=1600)).invoke(state)
    assert result['status'] == 'complete'
    coverage = coverage_summary(result)
    assert all(record['source_characters'] <= 1600 for record in coverage['roles'].values())
    assert coverage['roles']['reader']['characters_not_supplied'] > 0
    assert 'Source coverage' in result['final_output']
    assert len(result['requests']) == result['model_calls'] == 7
    assert all(request['status'] == 'responded' for request in result['requests'])
    assert result['requests'][-1]['role'] == 'reviewer'
    # Prepared packets from a failed provider request are not claimed as delivered.
    failing = Mock(api_key='mock-only')
    failing.chat.completions.create.side_effect = RuntimeError('provider internal')
    failed = build_graph(failing, 'model').invoke(document())
    assert failed['status'] == 'incomplete'
    assert failed['coverage']['reader']['evidence_ids'] == []
    assert failed['coverage']['reader']['preview_section_ids'] == []


def test_token_usage_is_provider_data_not_guessed():
    from types import SimpleNamespace as NS
    from llm import call_model
    client = Mock()
    response = HELPERS['response'](HELPERS['findings']())
    response.usage = NS(prompt_tokens=12, completion_tokens=7, total_tokens=19)
    client.chat.completions.create.return_value = response
    usage = []
    call_model(client, 'model', [], [], 128, False, usage_log=usage)
    assert usage == [{'prompt_tokens':12,'completion_tokens':7,'total_tokens':19}]
    assert 'tools' not in client.chat.completions.create.call_args.kwargs


def test_truncated_output_still_records_confirmed_source_delivery():
    client = Mock(api_key='mock-only')
    client.chat.completions.create.return_value = HELPERS['response']('partial', finish='length')
    result = build_graph(client, 'model').invoke(document())
    assert result['status'] == 'incomplete'
    assert result['coverage']['reader']['evidence_ids']
    assert result['requests'][0]['status'] == 'responded'


def test_tool_budget_omissions_are_explicit_and_do_not_truncate_passages():
    from reading import bound_tool_result
    state = document()
    entries = list(state['evidence'].values())[:4]
    result, supplied = bound_tool_result('search_paper', {'matches':entries, 'total_matches':4}, 800)
    assert result['omitted_by_context_limit'] > 0
    assert sum(len(entry['text']) for entry in supplied) <= 800
    assert all(entry == state['evidence'][entry['evidence_id']] for entry in supplied)
    unavailable, supplied = bound_tool_result('get_evidence', entries[0], 0)
    assert supplied == [] and 'text' not in unavailable
    assert 'limit' in unavailable['unavailable']

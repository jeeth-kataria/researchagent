import copy
import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from agents import Limits
from evaluate import (prepare_synthetic, load_cases, plan, run_case, score,
                      annotation_template, digest, token_totals)

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_workflow.py')))


def fixture_run(tmp_path):
    cases = load_cases(prepare_synthetic(tmp_path))
    client = Mock(api_key='mock-only')
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.side_effect = [HELPERS['response'](HELPERS['review']('paper'))] + HELPERS['complete_run'](HELPERS['review']('paper'))
    run = run_case(cases['cases'][0], 'mock-only', 'openai/gpt-oss-20b', Limits(),
                   client_factory=lambda _:client, execution_mode='mocked')
    return run, client


def human(run):
    annotations = annotation_template(run)
    annotations.update(assessor='Synthetic test annotator', human_attested=True)
    for arm in annotations['arms'].values():
        for item in arm:
            item.update(assessor='Synthetic test annotator', rationale='Mechanical scoring fixture')
    return annotations


def test_case_preparation_version_hashes_and_empty_real_corpus(tmp_path):
    document = load_cases(prepare_synthetic(tmp_path))
    estimate = plan(document, Limits(), 1)
    assert estimate['available_real_cases'] == 0
    assert estimate['available_synthetic_cases'] == 2
    assert estimate['maximum_model_requests'] == 13
    assert load_cases('evaluation/cases.json')['cases'] == []
    assert all(len(case['pdf_sha256']) == 64 for case in document['cases'])


def test_wrong_pdf_version_or_anchor_is_rejected(tmp_path):
    path = prepare_synthetic(tmp_path)
    document = json.loads(path.read_text())
    document['cases'][0]['pdf_sha256'] = '0'*64
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='hash'):
        load_cases(path)
    path = prepare_synthetic(tmp_path)
    document = json.loads(path.read_text())
    document['cases'][0]['known_issues'][0]['passages'][0]['quote'] = 'not present'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='passage'):
        load_cases(path)


def test_fair_baseline_full_text_same_model_and_preserved_inputs(tmp_path):
    result, client = fixture_run(tmp_path)
    assert result['status'] == 'complete' and result['execution_mode'] == 'mocked'
    assert result['arms']['baseline']['model_calls'] == 1
    assert result['arms']['workflow']['model_calls'] == 7
    assert result['arms']['baseline']['coverage']['full_extracted_text'] is True
    assert len(result['arms']['baseline']['coverage']['evidence_ids']) >= 2
    assert result['arms']['baseline']['usage']['total_tokens'] is None
    assert all(record['elapsed_seconds'] > 0 for record in result['arms'].values())
    assert result['source_sha256']['agents.py']
    assert len(result['arms']['workflow']['requests']) == 7
    assert all(call.kwargs['model'] == 'openai/gpt-oss-20b' for call in client.chat.completions.create.call_args_list)


def test_unknown_findings_are_unjudged_not_false_alarms(tmp_path):
    run, _ = fixture_run(tmp_path)
    output = score(run, human(run))
    for arm in output['arms'].values():
        assert arm['unjudged'] == 1 and arm['false_alarms'] == 0
        assert arm['missed_known_issues'] is None
        assert arm['known_issue_recall_percent'] is None
    assert output['real_cases_fully_scored'] == 0


def test_known_match_raw_denominators_and_recall(tmp_path):
    run, _ = fixture_run(tmp_path)
    annotations = human(run)
    for arm in annotations['arms'].values():
        arm[0].update(verdict='matches_known_issue', issue_ids=['unmatched-comparison'])
    scored = score(run, annotations)
    for result in scored['arms'].values():
        assert result['known_issues_identified'] == result['known_issues_total'] == 1
        assert result['known_issue_recall_percent'] == 100
        assert result['judged_unique_findings'] == 1 and result['missed_known_issues'] == 0
    assert scored['real_cases_fully_scored'] == 0  # Synthetic scores are never a real-paper benchmark.


def test_duplicates_are_not_double_credited_or_double_false_alarms(tmp_path):
    run, _ = fixture_run(tmp_path)
    for arm in run['arms'].values():
        arm['output']['concerns'] *= 2
    annotations = human(run)
    for arm in annotations['arms'].values():
        for item in arm:
            item.update(verdict='false_alarm')
    result = score(run, annotations)
    for arm in result['arms'].values():
        assert arm['duplicates'] == 1 and arm['false_alarms'] == 1
        assert arm['precision_percent'] == 0 and arm['judged_unique_findings'] == 1


@pytest.mark.parametrize('mutation', ['llm', 'no_attestation', 'wrong_fingerprint', 'unknown_issue'])
def test_invalid_or_nonhuman_annotations_are_rejected(tmp_path, mutation):
    run, _ = fixture_run(tmp_path)
    annotations = human(run)
    if mutation == 'llm':
        annotations['assessment_method'] = 'llm'
    elif mutation == 'no_attestation':
        annotations['human_attested'] = False
    elif mutation == 'wrong_fingerprint':
        annotations['arms']['baseline'][0]['finding_sha256'] = 'wrong'
    else:
        annotations['arms']['baseline'][0].update(verdict='matches_known_issue', issue_ids=['invented'])
    with pytest.raises(ValueError):
        score(run, annotations)


def test_quota_failure_stops_without_running_other_arm(tmp_path):
    from llm import RunError
    from unittest.mock import patch
    cases = load_cases(prepare_synthetic(tmp_path))
    client = Mock(api_key='mock-only')
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    with patch('evaluate.call_model', side_effect=RunError('Groq quota/rate limit reached.')):
        run = run_case(cases['cases'][0], 'mock-only', 'model', Limits(), client_factory=lambda _:client, execution_mode='mocked')
    assert run['status'] == 'partial' and set(run['arms']) == {'baseline'}
    assert run['arms']['baseline']['model_calls'] == 1
    assert run['arms']['baseline']['output'] == {}


def test_additional_valid_concerns_and_human_semantic_duplicates(tmp_path):
    run, _ = fixture_run(tmp_path)
    for arm in run['arms'].values():
        arm['output']['concerns'].append(dict(arm['output']['concerns'][0], concern='Same issue phrased differently.'))
    annotations = human(run)
    for arm in annotations['arms'].values():
        arm[0].update(verdict='additional_valid')
        arm[1].update(verdict='duplicate', duplicate_of=0)
    result = score(run, annotations)
    for arm in result['arms'].values():
        assert arm['additional_valid'] == arm['duplicates'] == 1
        assert arm['known_issues_identified'] == 0 and arm['missed_known_issues'] == 1
        assert arm['false_alarms'] == arm['unjudged'] == 0


def test_report_concerns_are_separate_from_paper_benchmark(tmp_path):
    run, _ = fixture_run(tmp_path)
    run['arms']['workflow']['output']['concerns'][0].update(target='report', affected_statement='A draft claim')
    result = score(run, human(run))
    assert result['arms']['workflow']['report_concerns'] == 1
    assert result['arms']['workflow']['judged_unique_findings'] == 0


def test_real_cases_require_confirmed_review_version(tmp_path):
    path = prepare_synthetic(tmp_path)
    document = json.loads(path.read_text())
    document['cases'][0]['kind'] = 'real'
    document['cases'][0]['known_issues'][0]['version_match_confirmed'] = False
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='same-version'):
        load_cases(path)


def test_full_text_baseline_preserves_whitespace_gaps():
    import re
    from evaluate import full_paper_context
    from workflow import initial_state
    text = 'First text.' + ' ' * 1700 + 'Last text.'
    state = initial_state('Read.', {1:text})
    full = full_paper_context(state)[0]['labeled_text']
    assert re.sub(r'\[SOURCE E\d+ / PDF page \d+\]', '', full) == text

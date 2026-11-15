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


def test_evaluation_only_text_override_is_explicit_and_bounded():
    from evaluate import case_text_limit
    assert case_text_limit({}) == 20000
    assert case_text_limit({'max_text_chars': 30000}) == 30000
    for value in (True, 0, 30001, '30000'):
        with pytest.raises(ValueError, match='text limit'):
            case_text_limit({'max_text_chars': value})


def public_fixture(tmp_path, monkeypatch):
    from evaluate import prepare_public
    manifest = prepare_synthetic(tmp_path / 'synthetic')
    case = json.loads(manifest.read_text())['cases'][0]
    pdf = Path(case['pdf']).read_bytes()
    # Keep relative names in a portable public manifest.
    case['pdf'] = 'paper.pdf'
    case['pdf_sha256'] = digest(pdf)
    review = json.dumps({'reviews': [{'comments': 'Independent reviewer issue.'}]}).encode()
    license_data = b'Public fixture licence'
    case.update(pdf_url='https://example.test/paper', review_url='https://example.test/review',
                license_url='https://example.test/license', review_sha256=digest(review),
                license_sha256=digest(license_data))
    for issue in case['known_issues']:
        issue.update(review_index=0, review_quote='Independent reviewer issue.')
    (tmp_path / 'evaluation').mkdir()
    (tmp_path / 'evaluation/public_cases.json').write_text(json.dumps({'cases': [case]}))
    monkeypatch.chdir(tmp_path)
    blobs = {case['pdf_url']: pdf, case['review_url']: review, case['license_url']: license_data}
    return prepare_public, blobs


def test_public_preparation_verifies_sources_and_reuses_local_cache(tmp_path, monkeypatch):
    prepare, blobs = public_fixture(tmp_path, monkeypatch)
    fetch = Mock(side_effect=blobs.__getitem__)
    manifest = prepare(tmp_path / 'downloaded', fetch)
    assert load_cases(manifest)['cases']
    assert fetch.call_count == 3
    prepare(tmp_path / 'downloaded', Mock(side_effect=AssertionError('unexpected network')))


def test_public_preparation_rejects_changed_review_bytes(tmp_path, monkeypatch):
    prepare, blobs = public_fixture(tmp_path, monkeypatch)
    blobs['https://example.test/review'] = b'changed review'
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare(tmp_path / 'downloaded', blobs.__getitem__)


def test_public_preparation_rejects_reference_quote_mismatch(tmp_path, monkeypatch):
    prepare, blobs = public_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'evaluation/public_cases.json'
    data = json.loads(path.read_text())
    data['cases'][0]['known_issues'][0]['review_quote'] = 'not in review'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Reference comment'):
        prepare(tmp_path / 'downloaded', blobs.__getitem__)


def test_guided_review_requires_explicit_human_attestation(tmp_path):
    from evaluate import review_interactively
    run, _ = fixture_run(tmp_path)
    answers = iter(['Test assessor', 'u', 'u', 'no'])
    output = tmp_path / 'annotations.json'
    assert review_interactively(run, output, lambda _: next(answers), lambda _: None) is None
    assert json.loads(output.read_text())['human_attested'] is False


def test_guided_review_records_real_human_choices_and_scores(tmp_path):
    from evaluate import review_interactively
    run, _ = fixture_run(tmp_path)
    # This is a mocked human session on a synthetic fixture, not a real scored paper.
    answers = iter(['Test assessor', 'm', 'unmatched-comparison', 'Fixture rationale',
                    'f', 'Second fixture rationale', 'YES'])
    output = tmp_path / 'annotations.json'
    scored = review_interactively(run, output, lambda _: next(answers), lambda _: None)
    assert scored['arms']['baseline']['known_issues_identified'] == 1
    assert scored['arms']['workflow']['false_alarms'] == 1
    assert scored['real_cases_fully_scored'] == 0
    assert json.loads(output.read_text())['human_attested'] is True


def test_guided_review_quit_preserves_unattested_draft(tmp_path):
    from evaluate import review_interactively
    run, _ = fixture_run(tmp_path)
    answers = iter(['Test assessor', 'q'])
    path = tmp_path / 'annotations.json'
    assert review_interactively(run, path, lambda _: next(answers), lambda _: None) is None
    assert json.loads(path.read_text())['human_attested'] is False


def test_guided_pilot_saves_runs_annotations_and_scores(tmp_path, monkeypatch):
    import sys
    import app
    import evaluate
    run, _ = fixture_run(tmp_path / 'fixtures')
    manifest = prepare_synthetic(tmp_path / 'cases')
    monkeypatch.setattr(app, 'settings', lambda: {'key': 'mock-only', 'model': 'mock-model', 'limits': Limits()})
    monkeypatch.setattr(evaluate, 'prepare_public', Mock(return_value=manifest))
    monkeypatch.setattr(evaluate, 'run_case', Mock(return_value=run))
    human_review = Mock(return_value=score(run, human(run)))
    monkeypatch.setattr(evaluate, 'review_interactively', human_review)
    out = tmp_path / 'runs'
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', 'pilot', '--live', '--max-calls', '13', '--out', str(out)])
    evaluate.main()
    assert human_review.call_count == 1
    assert list(out.glob('*.annotations.json')) and list(out.glob('*.scores.json'))
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['runs'][0]['status'] == 'complete'
    assert summary['real_cases_scored'] == 0  # This test used a synthetic case.


def test_cli_failure_saves_partial_output_and_exits_nonzero(tmp_path, monkeypatch):
    import sys
    import app
    import evaluate
    run, _ = fixture_run(tmp_path / 'fixtures')
    run['status'] = 'partial'
    run['arms']['workflow'].update(status='incomplete', error='Safe provider failure')
    manifest = prepare_synthetic(tmp_path / 'cases')
    monkeypatch.setattr(app, 'settings', lambda: {'key': 'mock-only', 'model': 'mock-model', 'limits': Limits()})
    fake_run = Mock(return_value=run)
    monkeypatch.setattr(evaluate, 'run_case', fake_run)
    out = tmp_path / 'runs'
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', 'run', '--cases', str(manifest),
        '--live', '--max-calls', '26', '--out', str(out)])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 1 and fake_run.call_count == 1
    assert list(out.glob('*.annotations.json'))
    assert json.loads((out / 'summary.json').read_text())['runs'][0]['status'] == 'partial'


def test_baseline_preserves_full_paper_without_json_escaping_pdf_lines(tmp_path):
    from evaluate import full_paper_context
    from pdf_reader import extract_pdf
    from workflow import initial_state
    result, client = fixture_run(tmp_path)
    state = initial_state(result['case']['question'], extract_pdf(Path(result['case']['pdf']).read_bytes()))
    user_message = client.chat.completions.create.call_args_list[0].kwargs['messages'][1]['content']
    for page in full_paper_context(state):
        assert page['labeled_text'] in user_message
        assert f"[PDF page {page['page']}]" in user_message
    assert result['case']['question'] in user_message


def test_evaluation_spacing_is_shared_across_clients_without_retries(monkeypatch):
    import evaluate
    now = [100.0]
    waits = []
    def wait(seconds):
        waits.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(evaluate.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(evaluate.time, 'sleep', wait)
    clients = []
    def client(key, http_client):
        clients.append(http_client)
        return http_client
    monkeypatch.setattr(evaluate, 'make_client', client)
    factory = evaluate.paced_client_factory(30)
    with factory('mock-only') as baseline_client, factory('mock-only') as workflow_client:
        baseline_client.event_hooks['request'][0](None)
        now[0] += 10
        workflow_client.event_hooks['request'][0](None)
    assert waits == [20]
    assert len(clients) == 2

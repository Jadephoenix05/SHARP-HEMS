"""Check every document against the project's decisions and the idea book.

Why this exists. Documents drift silently. A hardware spec that still says
"fan on PWM" after the decision that fans are never dimmed does not fail
anything - it just quietly sends someone to order the wrong parts, or to build a
UI control for an action that no longer exists.

Two sources of truth are checked, and they are different in kind:

  DECISIONS  - what the project lead has ruled. These override everything,
               including earlier drafts of these same documents.
  IDEA BOOK  - the vocabulary and taxonomy the project is built on. Documents
               must use the same four load classes, not invent their own.

A rule is either REQUIRED (the text must appear somewhere) or FORBIDDEN (it must
appear nowhere). Forbidden rules carry an exemption list, because a document is
allowed to say "we do not dim fans" - it is only forbidden to instruct someone
to dim one.
"""
from pathlib import Path
import argparse
import io
import json
import re
import sys

# The documents contain arrows and rupee signs; a Windows console defaults to
# cp1252 and would crash on them mid-report.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DOCS = Path('docs')

# Rulings from the project lead, in the order they were made. Each is stated as
# it was given, so a reader can audit the rule against the instruction.
DECISIONS = [
    {
        'id': 'D1',
        'ruling': 'Fans and lights are must - never shed.',
        'forbidden': [r'\bshed(?:ding)? (?:the )?(?:ceiling )?fan\b',
                      r'\bshed(?:ding)? (?:the )?light'],
        'required_in': {'HARDWARE_PROTOTYPE_SPEC.md': r'[Cc]ritical',
                        'DASHBOARD_SPECIFICATION.md': r'never shed'},
    },
    {
        'id': 'D2',
        'ruling': 'No dimming of fans and lights at all.',
        'forbidden': [r'fan.{0,40}PWM', r'PWM.{0,40}fan',
                      r'light.{0,30}must dim', r'DIM 50', r'dims a fan',
                      r'level 2 -> PWM', r'level 2 → PWM'],
        'exempt_phrases': ['never dim', 'not used on any critical',
                           'no PWM', 'left free', 'reintroduced',
                           'gives up', 'decision', 'stays in the contract'],
    },
    {
        'id': 'D3',
        'ruling': 'Use the four load classes from the idea book.',
        'required_in': {
            'HARDWARE_PROTOTYPE_SPEC.md': r'Critical.*Thermostatic|Thermostatic.*Deferrable',
            'DASHBOARD_SPECIFICATION.md': r'Critical.*Thermostatic|Thermostatic.*Deferrable',
        },
    },
    {
        'id': 'D4',
        'ruling': 'Never cut a household off; restrict luxury instead.',
        'required_in': {'DASHBOARD_SPECIFICATION.md': r'luxury|discretionary'},
    },
    {
        'id': 'D5',
        'ruling': 'The AC is on/off plus an ADVISORY setpoint, never automated.',
        'forbidden': [r'automat\w+ (?:the )?setpoint', r'sets the temperature'],
        'exempt_phrases': ['may NOT set', 'may not set', 'not automated',
                           'Advisory only', 'advisory', 'future work'],
        'required_in': {'DASHBOARD_SPECIFICATION.md': r'[Aa]dvisory'},
    },
    {
        'id': 'D6',
        'ruling': 'No time-of-day tariff exists; a surcharge must be labelled a proposal.',
        'forbidden': [r'time-of-day tariff (?:is|will be) applied'],
        'required_in': {'DASHBOARD_SPECIFICATION.md': r'no time-of-day|telescopic'},
    },
    {
        'id': 'D7',
        'ruling': 'The model is never presented as deployable.',
        'forbidden': [r'ready for deployment', r'approved for deployment(?!\W*(?:is|:)?\s*false)',
                      r'production[- ]ready'],
        'exempt_phrases': ['not', 'never', 'false'],
    },
    {
        'id': 'D8',
        'ruling': 'Target homes first: no solar, no inverter.',
        'required_in': {'DASHBOARD_SPECIFICATION.md': r'431|no inverter'},
    },
]

# Vocabulary the idea book defines. Documents must not invent synonyms for it.
IDEA_BOOK_TERMS = ['Critical', 'Thermostatic', 'Deferrable', 'Interruptible']


def scan(text, pattern, exempt):
    """Lines matching pattern that are not excused by an exemption phrase."""
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line, re.I):
            if any(phrase.lower() in line.lower() for phrase in exempt):
                continue
            hits.append((number, line.strip()[:110]))
    return hits


def validate(root):
    docs = {p.name: p.read_text(encoding='utf-8') for p in sorted((root / DOCS).glob('*.md'))}
    failures, warnings, checks = [], [], 0

    print(f'{len(docs)} documents\n')
    for decision in DECISIONS:
        exempt = decision.get('exempt_phrases', [])
        problems = []
        for pattern in decision.get('forbidden', []):
            for name, text in docs.items():
                for number, line in scan(text, pattern, exempt):
                    problems.append(f'{name}:{number}  {line}')
                checks += 1
        for name, pattern in decision.get('required_in', {}).items():
            checks += 1
            if name not in docs:
                problems.append(f'{name} is missing entirely')
            elif not re.search(pattern, docs[name], re.I | re.S):
                problems.append(f'{name} does not state it')

        mark = 'PASS' if not problems else 'FAIL'
        print(f"  [{mark}] {decision['id']}: {decision['ruling']}")
        for problem in problems[:4]:
            print(f'         {problem}')
        if problems:
            failures.append({'decision': decision['id'],
                             'ruling': decision['ruling'], 'problems': problems})

    print('\nIdea book vocabulary')
    for name in ['HARDWARE_PROTOTYPE_SPEC.md', 'DASHBOARD_SPECIFICATION.md']:
        checks += 1
        if name not in docs:
            continue
        missing = [t for t in IDEA_BOOK_TERMS if t.lower() not in docs[name].lower()]
        mark = 'PASS' if not missing else 'WARN'
        print(f'  [{mark}] {name}: '
              + ('all four classes named' if not missing
                 else f'does not name {missing}'))
        if missing:
            warnings.append({'document': name, 'missing_classes': missing})

    report = {
        'status': 'DOCS_CONSISTENT' if not failures else 'DOCS_INCONSISTENT',
        'documents': sorted(docs),
        'checks_run': checks,
        'decisions': [{'id': d['id'], 'ruling': d['ruling']} for d in DECISIONS],
        'failures': failures,
        'warnings': warnings,
        'note': ('Decisions override earlier drafts of these same documents. '
                 'A forbidden phrase is excused when the line is stating the '
                 'rule rather than instructing someone to break it.'),
    }
    (root / 'reports/doc_consistency_v1.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

    print(f"\n{'=' * 58}")
    print(f"{report['status']}  ({checks} checks, {len(failures)} failures, "
          f"{len(warnings)} warnings)")
    print(f"{'=' * 58}")
    return not failures


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    raise SystemExit(0 if validate(a.root.resolve()) else 1)

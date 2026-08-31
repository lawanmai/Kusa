# -*- coding: utf-8 -*-
"""Build a text-free release of the round 1 re-annotation labels.

Joins the private key (arm membership, gold label, category) with the two
annotator exports on the item id and writes id-level labels only. No sentence
from KurdiSent, and no annotator free-text note, enters the output: the corpus
is not ours to redistribute, and the notes may quote it.

Anyone holding KurdiSent can rejoin this file on the item id and reproduce every
agreement figure in the paper.
"""
import csv
import io
import os

KUSA = os.environ.get('KUSA_ROOT', r'G:\My Drive\kusa')
KEY = KUSA + r'\annotation\round1\round1_key_PRIVATE.csv'
EXP_A = KUSA + r'\annotation\round1\round1_export_annotatorA.csv'
EXP_B = KUSA + r'\annotation\round1\round1_export_annotatorB.csv'
OUT = KUSA + r'\kusa\annotation\round1_labels_public.csv'

csv.field_size_limit(10 ** 7)


def read(p):
    return list(csv.DictReader(io.open(p, encoding='utf-8-sig')))


key = {r['id']: r for r in read(KEY)}
lab_a = {r['id']: r['sentiment'] for r in read(EXP_A)}
lab_b = {r['id']: r['sentiment'] for r in read(EXP_B)}

FIELDS = ['id', 'category', 'arm', 'arm_eval', 'evaluated', 'is_audit',
          'gold_label', 'label_A', 'label_B']

rows = []
for ident in sorted(key):
    k = key[ident]
    rows.append({
        'id': ident,
        'category': k['category'],
        'arm': k['arm'],
        'arm_eval': k['arm_eval'],
        'evaluated': k['evaluated'],
        'is_audit': k['is_audit'],
        'gold_label': k['gold_name'],
        'label_A': lab_a.get(ident, ''),
        'label_B': lab_b.get(ident, ''),
    })

# hard guarantee: nothing but the agreed vocabulary leaves this script
ALLOWED = {'Negative', 'Neutral', 'Positive', ''}
for r in rows:
    assert r['label_A'] in ALLOWED, r['label_A']
    assert r['label_B'] in ALLOWED, r['label_B']
    assert r['gold_label'] in ALLOWED, r['gold_label']

with io.open(OUT, 'w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

n_eval = sum(1 for r in rows if r['evaluated'] in ('True', 'true', '1'))
print('wrote %s' % OUT)
print('  %d items, %d evaluated, %d audit-only'
      % (len(rows), n_eval, len(rows) - n_eval))
print('  columns: %s' % ', '.join(FIELDS))

import argparse
import torch
import os
import re
import glob

from nltk.translate import Alignment
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(description="Merge weights files for two directions")

    parser.add_argument("forward", help="forward weights")
    parser.add_argument("backward", help="backward weights")
    parser.add_argument("--bialign", help="bidirectional alignments")
    parser.add_argument("--ref", help="talp file")
    parser.add_argument("--output", help="output filename")
    parser.add_argument("--forward-suffix", default="f", help="output filename")
    parser.add_argument("--backward-suffix", default="b", help="output filename")
    parser.add_argument("--keep-common", action="store_true", help="output filename")

    return parser.parse_args()

def parse_refs(filename):
    refs = []
    poss = []
    for line in open(filename):
        line = line.strip()
        refs.append(Alignment.fromstring(re.sub(r'[0-9]*p[0-9]*', "", line)))
        poss.append(Alignment.fromstring(line.replace('p','-')))
    return refs, poss

def alignment_merics(hyps, refs, poss):
    n_common_ref = sum([len(hyp & ref) for hyp, ref in zip(hyps, refs)])
    n_common_pos = sum([len(hyp & pos) for hyp, pos in zip(hyps, poss)])
    n_hyps = sum([len(hyp) for hyp in hyps])
    n_refs = sum([len(ref) for ref in refs])
    precision = n_common_pos / float(n_hyps)
    recall = n_common_ref / float(n_refs)
    aer = 1.0 - (n_common_ref + n_common_pos) / float(n_hyps + n_refs)
    return aer, precision, recall

def merge(tokens):
    n = len(tokens)
    group = []
    res = []
    cnt = -1
    if tokens[0].startswith('▁'):
        # "▁你 好 ▁啊"
        for i in range(n):
            if tokens[i].startswith('▁'):
                res.append(tokens[i].replace('▁',''))
                cnt += 1
                group.append(cnt)
            else:
                res[-1] = res[-1] + tokens[i]
                group.append(cnt)
    else:
        # "你@@ 好 啊"
        last_flag = False
        for i in range(n):
            if tokens[i].endswith('@@'):
                cur_flag = True
                tok = tokens[i].replace('@@','')
            else:
                cur_flag = False
                tok = tokens[i]
            if last_flag:
                res[-1] = res[-1] + tok
                group.append(cnt)
            else:
                res.append(tok)
                cnt += 1
                group.append(cnt)
            last_flag = cur_flag
    reverse_group = [[] for i in range(max(group)+1)]
    for i in range(len(group)):
        reverse_group[group[i]].append(i)
    return res, group, reverse_group

def align_to_weights(ref, pos, src, tgt):
    """
    ref: Alignment
    pos: Alignment
    src: bpe tokens
    tgt: bpe tokens
    """
    t = []
    _, _, src_r = merge(src)
    _, _, tgt_r = merge(tgt)
    for x, y in ref & pos:
        for xx, yy in product(src_r[x-1], tgt_r[y-1]):
            t.append([xx, yy, 1])
    for x, y in pos - ref:
        for xx, yy in product(src_r[x-1], tgt_r[y-1]):
            t.append([xx, yy, 0.5])
    return t

def label_name(path):
    names = path.split("/")
    if 'exp' not in path:
        return names[0]
    idx = names.index('exp')
    return names[idx+1]

def merge_dict(src, tgt, args):
    res = {}
    for k, v in src.items():
        res[k] = v
    for k, v in tgt.items():
        if k not in res:
            res[k] = v
        elif args.keep_common:
            src_k = k + '_' + args.forward_suffix
            tgt_k = k + '_' + args.backward_suffix
            res[src_k] = res.pop(k)
            res[tgt_k] = v
    return res

def main(args):
    forward = torch.load(args.forward, map_location='cpu') # list of ['src', 'tgt', 'weights', 'metricss']
    backward = torch.load(args.backward, map_location='cpu') # list of ['src', 'tgt', 'weights', 'metricss']
    assert len(forward) == len(backward)
    res = []
    if args.bialign is not None:
        assert args.ref is not None
        refs, poss = parse_refs(args.ref)
        bi_aligns = [Alignment.fromstring(line.strip()) for line in open(args.bialign)]
        bi_metrics = [alignment_merics([hyp], [ref], [pos]) for hyp, ref, pos in zip(bi_aligns, refs, poss)]
        assert len(forward) == len(backward) == len(bi_aligns) == len(bi_metrics)
        for f, b, bi_align, bi_metric in zip(forward, backward, bi_aligns, bi_metrics):
            res_t = {}
            assert f['src'] == b['src'] and f['tgt'] == b['tgt']
            res_t['src'] = f['src']
            res_t['tgt'] = f['tgt']
            res_t['weights'] = merge_dict(f['weights'], b['weights'], args)
            res_t['metrics'] = merge_dict(f['metrics'], b['metrics'], args)
            res_t['weights']['bi_align'] = align_to_weights(bi_align, bi_align, f['src'], f['tgt'])
            res_t['metrics']['bi_align'] = bi_metric
            res.append(res_t)
    else:
        for f, b in zip(forward, backward):
            res_t = {}
            res_t['src'] = f['src']
            res_t['tgt'] = f['tgt']
            res_t['weights'] = merge_dict(f['weights'], b['weights'], args)
            res_t['metrics'] = merge_dict(f['metrics'], b['metrics'], args)
            res.append(res_t)

    output = args.output or args.forward
    torch.save(res, output)


if __name__ == "__main__":
    main(parse_args())
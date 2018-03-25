#!/bin/python3
import argparse
from collections import defaultdict
from itertools import combinations, chain

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("support", help="minimum support", type=int)
parser.add_argument("input", help="input file name", type=str)
parser.add_argument("output", help="output file name", type=str)
args = parser.parse_args()

def apriori(data, support):
    def scan(data, candidates, support):
        count = defaultdict(int)
        for transaction in data:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    count[tuple(sorted(candidate))] += 1
        return {k: v/len(data) for k, v in count.items() if v/len(data) > support}
    result = dict()
    k = 2
    candidates = list(map(lambda x: set([x]), set(chain(*data))))
    while candidates:
        filtered = scan(data, candidates, support)
        result.update(filtered)
        candidates = list(map(set, combinations(set(chain(*filtered.keys())), k)))
        k += 1
    return result

def createRule(freq):
    def confidence(conseq, seq):
        return round(freq[conseq] / freq[tuple(sorted(set(conseq) - set([seq])))] * 100, 2)
    def support(conseq):
        return round(freq[conseq] * 100, 2)
    
    for conseq, v in filter(lambda item: len(item[0]) > 1, freq.items()):
        for seq in conseq:
            yield set(conseq) - set([seq]), set([seq]), support(conseq), confidence(conseq, seq)

with open(args.input) as f:
    data = [line.split() for line in f.readlines()]

freq = apriori(data, args.support/100)

with open(args.output, 'w') as f:
    for item, associative, support, confidence in createRule(freq):
        f.write('{{{}}}\t{{{}}}\t{:.2f}\t{:.2f}\n'.format(','.join(item), ','.join(associative), support, confidence))

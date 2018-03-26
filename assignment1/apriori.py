#!/bin/python3
import argparse
from collections import defaultdict
from itertools import combinations, chain

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("support", help="minimum support", default=.0, type=float)
parser.add_argument("input", help="input file name", type=str)
parser.add_argument("output", help="output file name", type=str)
parser.add_argument("confidence", help="minimum confidence", nargs='?', const=1, default=.0, type=float)
args = parser.parse_args()

def apriori(data, support):
    candidates = list(map(lambda x: frozenset([x]), set(chain(*data))))
    result = dict()
    k = 2
    def scan():
        count = defaultdict(int)
        for transaction in data:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    count[candidate] += 1
        return {k: v/len(data) for k, v in count.items() if float(v)/len(data) >= support}
    while candidates:
        filtered = scan()
        result[k - 1] = filtered
        candidates = set([i.union(j) for i in filtered.keys() for j in filtered.keys() if len(i.union(j)) == k])
        k += 1
    return result

def define(freq, transactions, confidence=.0):
    def support(item):
        return float(freq[len(item)][item])/len(transactions)
    
    for k, v in freq.items():
        if k == 1: continue
        for item in v:
            for element in map(frozenset, chain(*[combinations(item, i) for i, e in enumerate(item, 1)])):
                remain = item.difference(element)
                if not remain: continue
                conf = support(item) / support(element)
                if conf >= confidence:
                    yield element, remain, round(freq[len(item)][item] * 100, 2), round(conf * 100, 2)

with open(args.input) as f:
    data = [line.split() for line in f.readlines()]

freq = apriori(data, args.support/100)
rules = define(freq, data, args.confidence)

with open(args.output, 'w') as f:
    for item, asso, sup, conf in rules:
        f.write('{{{}}}\t{{{}}}\t{:.2f}\t{:.2f}\n'.format(','.join(map(str, item)), ','.join(map(str, asso)), sup, conf))

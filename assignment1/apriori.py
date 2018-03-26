#!/bin/python3
import argparse
from collections import defaultdict
from itertools import combinations, chain
from decimal import Decimal, getcontext, ROUND_HALF_UP

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("support", help="minimum support", default=.0, type=float)
parser.add_argument("input", help="input file name", type=str)
parser.add_argument("output", help="output file name", type=str)
parser.add_argument("confidence", help="minimum confidence", nargs='?', const=1, default=.0, type=float)
args = parser.parse_args()

# set decimal context as ROUND_HALF_UP
getcontext().rounding=ROUND_HALF_UP

# Algorithm apriori
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

# define rules from frequency map
def define(freq, transactions, confidence=.0):
    f = lambda item: freq[len(item)][item]
    r = lambda value, k: type(value)(round(Decimal(value), k))
    for k, v in freq.items():
        if k == 1: continue
        for item in v:
            for element in map(frozenset, chain(*[combinations(item, i) for i, e in enumerate(item, 1)])):
                remain = item.difference(element)
                if not remain: continue
                (a, b), (c, d) = f(item).as_integer_ratio(), f(element).as_integer_ratio()
                conf = (a / b) / (c / d)
                if conf >= confidence:
                    yield element, remain, r(f(item) * 100, 2), r(conf * 100, 2)

if __name__ == '__main__':
    with open(args.input) as f:
        data = [list(map(int, line.split())) for line in f.readlines()]

    freq = apriori(data, args.support/100)
    rules = define(freq, data, args.confidence)

    with open(args.output, 'w') as f:
        for item, asso, sup, conf in rules:
            f.write('{{{}}}\t{{{}}}\t{:.2f}\t{:.2f}\n'.format(','.join(map(str, item)), ','.join(map(str, asso)), sup, conf))

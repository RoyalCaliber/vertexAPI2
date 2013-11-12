#!/usr/bin/python3
#script to compare integer distance output from sssp and bfs

import sys


def readLine(f):
  line = f.readline()
  if line:
    return tuple(int(x) for x in line.strip().split())
  else:
    return None


#compares the sorted test and gold files
def mergeCmp(fTest, fGold):
  lTest = readLine(fTest)
  lGold = readLine(fGold)
  while lGold is not None:
    while lTest[0] < lGold[0]:
      lTest = readLine(fTest)
    if lTest is None or lTest[0] != lGold[0]:
      print('vertex %d is in gold but not in test' % lGold[0])
      return False
    if lTest[1] != lGold[1]:
      if not (lGold[1] == 10000000 and lTest[1] == 2147383647):
        print('%d gold=%d ref=%d' % (lGold[0], lGold[1], lTest[1]))
        return False
    lTest = readLine(fTest)
    lGold = readLine(fGold)
  return True
  

if __name__ == '__main__':
  try:
    testFn, goldFn = sys.argv[1:]
  except ValueError:
    print("Usage: checkDist.py test gold")
    sys.exit(1)

  if mergeCmp(open(testFn), open(goldFn)):
    print("No differences found")
    sys.exit(0)
  else:
    sys.exit(1)

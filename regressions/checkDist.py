#!/usr/bin/python3

#########################################################################
#Copyright 2013 Royal Caliber LLC. (http://www.royal-caliber.com)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#########################################################################

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

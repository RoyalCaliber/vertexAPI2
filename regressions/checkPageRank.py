#!/usr/bin/python2

############################################################################
#Copyright 2013 SYSTAP, LLC. http://www.systap.com
#
#Written by Erich Elsen and Vishal Vaidyanathan
#of Royal Caliber, LLC
#Contact us at: info@royal-caliber.com
#
#This file was taken from mpgraph v0.1 which was (partially) funded by the
#DARPA XDATA program under AFRL Contract #FA8750-13-C-0002.
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
###########################################################################

#script to compare pagerank outputs

import sys
from math import fabs

#load ranks for file fn
def load( f ):
  ret = {}
  for line in f:
    vid, val = line.strip().split()
    ret[ int(vid) ] = float(val)
  return ret


def compare( tol_vals, tol_allowed, test, gold ):
  histo_counts = [0] * (len(tol_vals) + 1)
  for vid, val in test.items():
    try:
      diff = fabs( gold[ vid ] - val )
      pos = len(tol_vals) - 1
      while pos >= 0 and diff < tol_vals[pos]:
        pos -= 1

      histo_counts[pos + 1] += 1
    except KeyError:
      print "vid ", vid, " is in test but not in gold"
	  #this is not an error, we just output all vertices
	  #but powergraph does not
      #return False

  totalItems = float(len(test))

  for idx in range(len(histo_counts)):
    histo_counts[idx] /= totalItems
    if histo_counts[idx] > tol_allowed[idx]:
      print "Percentage too high: ", tol_allowed[idx], histo_counts[idx]
      return False

  return True


if __name__ == '__main__':
  if len( sys.argv ) != 3:
    print "Usage: checkPageRank.py test gold"
    sys.exit(1)

  test = sys.argv[1]
  gold = sys.argv[2]

  td = load( open(test) )
  gd = load( open(gold) )
  #this means we allow up to 100% of values differing by less than .0001
  #.9% of values by more than .0001 and less than .001
  #.09% of values by more than .001 and less than .01
  #.009% of values by more than .01 and less than .1
  #0 values more than .1
  if not compare( [.0001, .001, .01, .1, 1, 10], [1., 1e-2, 5e-3, 5e-4, 5e-5, 5e-6, 0], td, gd ):
    sys.exit(1)

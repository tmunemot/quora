#!/usr/bin/python
import sys
import os
import time
import multiprocessing

def notify(section, dlen=75):
    if notify.t_start != -1:
        # print time elapsed
        t = time.time()
        msg = str(t - notify.t_prev) + " sec (" + str(t - notify.t_start) + ")"
        notify.t_prev = t
        print msg
    else:
        # initialize time
        notify.t_start = time.time()
        notify.t_prev = notify.t_start
    msg = "section {0}, {1}".format(notify.section_id, section)
    print msg + '-' * (dlen - len(msg))
    notify.section_id += 1
notify.t_start = -1
notify.section_id = 1

def indices_for(df, nprocs):
	N = df.shape[0]
	L = int(N / nprocs)
	indices = []
	for i in range(nprocs):
		for j in range(L):
			indices.append(i)
	for i in range(N - (nprocs * L)):
		indices.append(nprocs - 1)
	return indices

def main(argv):
    print "do not run"

if __name__ == '__main__':
    exit(main(sys.argv[1:]))

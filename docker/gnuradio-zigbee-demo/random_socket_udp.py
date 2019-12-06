import socket
import sys, select, os
import random
import time
import argparse

parser = argparse.ArgumentParser(description='Generate random zigbee packets.')
parser.add_argument('size', metavar='S', type=int, nargs=1,
                   help='The size to transfert in byte')
args = parser.parse_args()

UDP_IP = "127.0.0.1"
UDP_PORT = 52001

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

start = time.time()
sumsize = 0
while sumsize < args.size[0]:

    if len(args.size) > 0 and sumsize >= args.size[0]:
        break;

    size = random.randrange(64)
    sumsize = sumsize + size
    msg = bytearray(random.getrandbits(8) for _ in range(size))
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        print("Exiting")
        line = input()
        break

stop = time.time()
elapsed = stop - start

#print(sumsize/elapsed)

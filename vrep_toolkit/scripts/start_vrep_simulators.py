import os
import subprocess
import argparse

VREP_DIR = '/home/colin/software/V-REP_PRO_EDU_V3_5_0_Linux/'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('simulation', type=str,
      help='The path to the simulation to load on startup')
  parser.add_argument('num_simulators', type=int,
      help='Number of V-Rep simulators to start')
  parser.add_argument('--h', default=False, action='store_true',
      help='Use this flag to run V-Rep in headless mode')
  parser.add_argument('--port', type=int, default=19997,
      help='The port to launch the simulator with. If using mulitple simulators this will be the first port set.')

  args = parser.parse_args()

  ports = range(args.port, args.port+args.num_simulators)
  simulation = os.path.abspath(args.simulation)
  if args.h:
    commands = ['./vrep.sh {} -gREMOTEAPISERVERSERVICE_{}_FALSE_FALSE -h'.format(simulation, port) for port in ports]
  else:
    commands = ['./vrep.sh {}  -gREMOTEAPISERVERSERVICE_{}_FALSE_FALSE'.format(simulation, port) for port in ports]

  os.chdir(VREP_DIR)
  for process in [subprocess.Popen(['/bin/bash', '-c', command]) for command in commands]:
    out, err = process.communicate()

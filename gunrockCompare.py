import subprocess

gunrock_executable = '/home/erich/gunrock/tests/bfs/bin/test_bfs_5.5_x86_64'
gunrock_options = '--device=1 --quick'

gas_executable = '/home/erich/vertexAPI2/bfs'
gas_options = '-m'

base_graph_dir = '/home/erich/gunrock/dataset/large/'
graphs = ['delaunay_n13', 'delaunay_n21', 'ak2010', 'coAuthorsDBLP', 'belgium_osm', 'webbase-1M', 'soc-LiveJournal1', 'kron_g500-logn21']

for graph in graphs:
  full_path = base_graph_dir + graph + '/' + graph + '.mtx'
  #run gas version first and grab starting vertex from output
  gas_output = subprocess.check_output([gas_executable, gas_options, full_path, '0'])
  vertex = gas_output.split()[2]
  gas_time = gas_output.split()[-2]

  #we need to know if the graph is symmetric since gunrock ignores that directive...
  f = open(full_path, 'r')
  firstline = f.readline()
  if 'symmetric' in firstline:
    undirected = '1'
  else:
    undirected = '0'

  #now run gunrock with the same starting vertex
  gunrock_output = subprocess.check_output([gunrock_executable, 'market', full_path, '--device=1', '--quick', '--src=' + vertex, '--undirected=' + undirected])
  fieldIsTime = False
  gunrock_time = ''
  for field in gunrock_output.split():
    if (fieldIsTime):
      gunrock_time = field
      break
    elif (field == 'elapsed:'):
      fieldIsTime = True

  print graph, gas_time, gunrock_time

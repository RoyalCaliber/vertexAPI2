/**
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include <graphlab.hpp>
#include <graphlab/util/stl_util.hpp>
#include <graphlab/macros_def.hpp>

#include "graphio.h"

/**
 * \brief The type used to measure distances in the graph.
 */
typedef int distance_type;

typedef int vertex_data;
typedef int edge_data;

int startVertex;

/**
 * \brief The graph type encodes the distances between vertices and
 * edges
 */
typedef graphlab::distributed_graph<vertex_data, edge_data> graph_type;


void init_vertex(graph_type::vertex_type& vertex)
{
  if (vertex.id() != startVertex)
    vertex.data() = 10000000;
  else
    vertex.data() = 0;
}

struct gather_type : graphlab::IS_POD_TYPE {
  int dist;
  gather_type(int dist = 10000000) : dist(dist) {}

  gather_type& operator+=(const gather_type& other) {
    dist = std::min(dist, other.dist);
    return *this;
  }
};

/**
 * \brief The single source shortest path vertex program.
 */
class sssp :
  public graphlab::ivertex_program<graph_type, gather_type>  {
  int min_dist;
  bool changed;
public:


  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::IN_EDGES;
  }; // end of gather_edges


  // /**
  //  * \brief Collect the distance to the neighbor
  //  */
   gather_type gather(icontext_type& context, const vertex_type& vertex,
              edge_type& edge) const {
     return gather_type(edge.data() + edge.source().data());
   } // end of gather function


  /**
   * \brief If the distance is smaller then update
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& min_dist) {
    int newVertexVal = std::min(vertex.data(), min_dist.dist);
    if (min_dist.dist == vertex.data())
      changed = false;
    else
      changed = true;

    vertex.data() = newVertexVal;
  }

  /**
   * \brief Determine if SSSP should run on all edges or just in edges
   */
  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    if(changed)
      return graphlab::OUT_EDGES;
    else
      return graphlab::NO_EDGES;
  }; // end of scatter_edges

  /**
   * \brief The scatter function just signal adjacent pages
   */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    context.signal(edge.target());
  } // end of scatter

  void save(graphlab::oarchive& oarc) const {
    oarc << min_dist;
  }

  void load(graphlab::iarchive& iarc) {
    iarc >> min_dist;
  }

}; // end of shortest path vertex program




/**
 * \brief We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", pagerank_writer()) to save the graph.
 */
struct shortest_path_writer {
  std::string save_vertex(const graph_type::vertex_type& vtx) {
    std::stringstream strm;
    strm << vtx.id() << "\t" << vtx.data() << "\n";
    return strm.str();
  }
  std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of shortest_path_writer



int main(int argc, char** argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_INFO);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options
    clopts("Single Source Shortest Path Algorithm.");
  std::string graph_dir;
  std::string format = "snap";
  std::string exec_type = "synchronous";
  size_t powerlaw = 0;
  std::vector<graphlab::vertex_id_type> sources;
  bool max_degree_source = false;
  clopts.attach_option("graph", graph_dir,
                       "The graph file.  If none is provided "
                       "then a toy graph will be created");
  clopts.add_positional("graph");

  clopts.attach_option("source", sources,
                       "The source vertices");
  clopts.attach_option("max_degree_source", max_degree_source,
                       "Add the vertex with maximum degree as a source");

  clopts.add_positional("source");

  clopts.attach_option("engine", exec_type,
                       "The engine type synchronous or asynchronous");


  clopts.attach_option("powerlaw", powerlaw,
                       "Generate a synthetic powerlaw out-degree graph. ");
  std::string saveprefix;
  clopts.attach_option("saveprefix", saveprefix,
                       "If set, will save the resultant pagerank to a "
                       "sequence of files with prefix saveprefix");

  if(!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }


  // Build the graph ----------------------------------------------------------
  graph_type graph(dc, clopts);
  {
    std::vector<int> h_edge_src_vertex;
    std::vector<int> h_edge_dst_vertex;
    std::vector<int> h_edge_values;
    int numVertices;
    if(powerlaw > 0) { // make a synthetic graph
      dc.cout() << "Loading synthetic Powerlaw graph." << std::endl;
      graph.load_synthetic_powerlaw(powerlaw, false, 2, 100000000);
    } else if (graph_dir.length() > 0) { // Load the graph from a file
      dc.cout() << "Loading graph in format: "<< format << std::endl;
      loadGraph( graph_dir.c_str(), numVertices, h_edge_src_vertex, h_edge_dst_vertex, &h_edge_values);
    } else {
      dc.cout() << "graph or powerlaw option must be specified" << std::endl;
      clopts.print_description();
      return EXIT_FAILURE;
    }

    std::vector<int> h_out_edges(numVertices);
    for (int i = 0; i < h_edge_src_vertex.size(); ++i) {
      graph.add_edge(h_edge_src_vertex[i], h_edge_dst_vertex[i], h_edge_values[i]);
      h_out_edges[h_edge_src_vertex[i]]++;
    }
    startVertex = std::max_element(h_out_edges.begin(), h_out_edges.end()) - h_out_edges.begin();
  }
  // must call finalize before querying the graph
  graph.finalize();
  dc.cout() << "#vertices:  " << graph.num_vertices() << std::endl
            << "#edges:     " << graph.num_edges() << std::endl;

  //find max out degree to start from
  //in case of tie, must find FIRST one
  sources.push_back(startVertex);

  graph.transform_vertices(init_vertex);
  // Running The Engine -------------------------------------------------------
  graphlab::omni_engine<sssp> engine(dc, graph, exec_type, clopts);


  // Signal all the vertices in the source set
  for(size_t i = 0; i < sources.size(); ++i) {
    engine.signal(sources[i]);
  }

  engine.start();
  const float runtime = engine.elapsed_seconds();
  dc.cout() << "Finished Running engine in " << runtime * 1000.
            << " milli-seconds." << std::endl;


  // Save the final graph -----------------------------------------------------
  if (saveprefix != "") {
    graph.save(saveprefix, shortest_path_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
  }

  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
} // End of main

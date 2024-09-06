import pydot

#!gprof2dot -f pstats agent.prof -o callingGraph.dot
(graph,) = pydot.graph_from_dot_file('callingGraph.dot')
graph.write_png('callingGraph.png')

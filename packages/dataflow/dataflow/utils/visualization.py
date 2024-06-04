'''Support for Graphviz visualization.'''

def weighted_graph_to_graph_viz(gr):
    '''Convert a weighted graph to graphviz.'''

    def compile_graph(gr):
        return ("\n".join([compile_state(s) for s in gr.nodes()])) + \
            ("\n".join([compile_edge(gr, e) for e in gr.edges()]))

    def compile_edge(gr, e):
        return f"\"{e[0]}\" -> \"{e[1]}\" [minlen=3 len=3 xlabel=\"{gr.edge_weight(e)}\"]"

    def compile_state(s):
        return f"\"{s}\""


    return f"""
		digraph PrecedenceGraph {{
		    graph [bgcolor=transparent,overlap=false]
		    node [fontsize=20 fontname="Calibri" width=0.6 penwidth=2 style=filled shape=circle]
		    edge[fontsize=16 fontname="Calibri"]
			{compile_graph(gr)}
		}}
        """

def weightedGraphToGraphViz(gr):

    def compileGraph(gr):
        return ("\n".join([compileState(s) for s in gr.nodes()])) + ("\n".join([compileEdge(gr, e) for e in gr.edges()]))

    def compileEdge(gr, e):
        return "\"{}\" -> \"{}\" [minlen=3 len=3 xlabel=\"{}\"]".format(e[0], e[1], gr.edge_weight(e))

    def compileState(s):
        return "\"{}\"".format(s)


    return """
		digraph PrecedenceGraph {{
		    graph [bgcolor=transparent,overlap=false]
		    node [fontsize=20 fontname="Calibri" width=0.6 penwidth=2 style=filled shape=circle]
		    edge[fontsize=16 fontname="Calibri"]
			{}
		}}
        """.format(compileGraph(gr))



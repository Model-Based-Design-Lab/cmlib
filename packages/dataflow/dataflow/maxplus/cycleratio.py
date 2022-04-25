	# mmcycle_robust ()
	#
	# Determines maximum ratio cycle in a directed graph in a robust way
	# G = (V, E, c), c: E->IR a "cost" function on the
	# edges, alternatively called "length" or "weight".
	# It has been observed that the mmcycle algorithm can behave non - deterministically
	# across compilers, when the graph has multiple equally critical cycles
	# Due to the order in which floating point calculations are scheduled, the
	# algorithm's control flow may follow different paths and lead to different
	# critical cycles as the output.
	# This function mmcycle_robust implements a version of the algorithm that is
	# deterministic across compilers. It may be a little slower.
	# For more info about the algorithm, see the mmcycle function.

    # typedef struct Graph
    # {
    #     long    n_nodes;
    #     node    *nodes;
    #     long    n_arcs;
    #     arc     *arcs;
    #     node    *vs;  // additional node with zero cost outgoing edges
    # } graph;


MCR_EPSILON_RATIO = 1.0e-8

def mmcycle_robust(Graph gr):

    h = DHeap()

    # set up initial tree
    tree = dict()
    sptr  = gr.vs()
    tree[sptr] = dict()
    tree[sptr]["in_list"] = false
    tree[sptr]["cost_t"] = 0.0
    tree[sptr]["transit_time_t"] = 0.0
    tree[sptr]["level"] = 0
    tree[sptr]["parent_in"] = None
    tree[sptr]["left_sibl"] = sptr
    tree[sptr]["right_sibl"] = sptr

    aptr = sptr.first_arc_out()
    vptr = aptr.head()
    tree[sptr]["first_child"] = vptr


	void mmcycle_robust(graph* gr, double* lambda, arc** cycle, long* len)
	{
		# const double MCR_EPSILON_RATIO = 1.0e-8L;
		# double min, infty, akey;
		# arc* aptr, * par_aptr, * vmin_aptr, * min_aptr;
		# node* sptr, * uptr, * vptr, * wptr;
		# bool foundCycle;
		# d_heap h;

		// set up initial tree
		sptr = gr->vs;
		sptr->in_list = false;
		sptr->cost_t = 0.0L;
		sptr->transit_time_t = 0.0L;
		sptr->level = 0L;
		sptr->parent_in = NILA;
		sptr->left_sibl = sptr;
		sptr->right_sibl = sptr;
		aptr = sptr->first_arc_out;
		vptr = aptr->head;
		sptr->first_child = vptr;
		while (aptr != NILA)
		{
			wptr = aptr->head;
			wptr->cost_t = 0.0L;
			wptr->transit_time_t = 0.0L;
			wptr->level = 1L;
			wptr->parent_in = aptr;
			wptr->first_child = NILN;
			wptr->left_sibl = vptr;
			vptr->right_sibl = wptr;
			wptr->in_list = false;
			aptr->in_tree = true;
			aptr->hpos = -1;  // arc does not go into heap
			aptr->key = DBL_MAX;
			vptr = wptr;
			aptr = aptr->next_out;
		}
		sptr->first_child->left_sibl = vptr;
		vptr->right_sibl = sptr->first_child;


		// determine upper bound on lambda, can be used as 'infinity'
		// adds up all costs, and divide by smallest non-zero transit time
		// requires that there are no cycles with zero transit time!
		// Also, determine epsilon value on transit times, cost and cost/time ratios
		// as a constant fraction of the smallest observed values
		double total_cost_plus_one = 1.0L;
		double min_transit_time = DBL_MAX;
		double min_cost = DBL_MAX;
		for (aptr = &(gr->arcs[gr->n_arcs - 1L]); aptr >= gr->arcs; aptr--)
		{
			// add costs to total cost
			total_cost_plus_one += fabs(aptr->cost);
			// keep min of transit times
			if (aptr->transit_time > 0.0L) {
				if (aptr->transit_time < min_transit_time) {
					min_transit_time = aptr->transit_time;
				}
			}
			// keep min of costs
			if (aptr->cost > 0.0L) {
				if (aptr->cost < min_cost) {
					min_cost = aptr->cost;
				}
			}
		}
		infty = total_cost_plus_one / min_transit_time;
		double epsilon_transit_time = MCR_EPSILON_RATIO * min_transit_time;
		double epsilon_cost_time_ratio = MCR_EPSILON_RATIO * (min_cost / min_transit_time);

		// initial keys of non tree edges are equal to arc costs
		for (aptr = &(gr->arcs[gr->n_arcs - 1L]); aptr >= gr->arcs; aptr--)
		{
			if (aptr->transit_time > epsilon_transit_time) {
				aptr->key = aptr->cost / aptr->transit_time;
			}
			else {
				aptr->key = infty;
			}
			aptr->in_tree = false;
		}



		// d-heap used for maintenance of vertex keys
		if (!ALLOC_HEAP(&h, gr->n_nodes))
			throw CException("Failed allocating heap");

		// compute initial vertex keys
		for (vptr = &(gr->nodes[gr->n_nodes - 1L]); vptr >= gr->nodes; vptr--)
		{
			min = DBL_MAX;
			vmin_aptr = NILA;
			aptr = vptr->first_arc_in;
			while (aptr != NILA)
			{
				if (!aptr->in_tree && (min - aptr->key > epsilon_cost_time_ratio))
				{
					min = aptr->key;
					vmin_aptr = aptr;
				}
				aptr = aptr->next_in;
			}
			vptr->vkey = vmin_aptr;
			if (vmin_aptr != NILA) {
				INSERT_ROBUST(&h, vmin_aptr, epsilon_cost_time_ratio);
			}
		}
		gr->vs->vkey = NILA;

		while (true)
		{
			min_aptr = GET_MIN(&h);
			ASSERT(min_aptr != NILA, "No element on heap!");

			*lambda = min_aptr->key;
			if (*lambda >= infty)
			{
				min_aptr = NILA;
				break; // input graph is acyclic in this case
			}

			uptr = min_aptr->tail;
			vptr = min_aptr->head;

			/* check if *vptr is an ancestor of *uptr in tree */

			foundCycle = false;
			par_aptr = uptr->parent_in;

			// MG: below is a fix, not in the original algorithm, since the original algorithm
			// does not seem to anticipate the possibility of self-edges in the graph.
			if (uptr == vptr)
			{
				// statement below was added to makje the calculation of the critical cycle work
				// correctly for critical self-edges.
				uptr->parent_in = min_aptr;
				break;
			}

			while (par_aptr != NILA)
			{
				if (par_aptr->tail == vptr)
				{
					foundCycle = true;
					break;
				}
				else
					par_aptr = par_aptr->tail->parent_in;
			}
			if (foundCycle) break;

			// it is not, remove edge (parent(v),v) from tree and make edge (u,v) a
			// tree edge instead
			par_aptr = vptr->parent_in;
			par_aptr->in_tree = false;
			min_aptr->in_tree = true;

			vptr->cost_t = uptr->cost_t + min_aptr->cost;
			vptr->transit_time_t = uptr->transit_time_t + min_aptr->transit_time;
			wptr = par_aptr->tail;

			// delete link (wptr,vptr) from tree
			if (vptr->right_sibl == vptr) {
				wptr->first_child = NILN;
			}
			else
			{
				vptr->right_sibl->left_sibl = vptr->left_sibl;
				vptr->left_sibl->right_sibl = vptr->right_sibl;
				if (wptr->first_child == vptr) {
					wptr->first_child = vptr->right_sibl;
				}
			}

			// insert link (uptr,vptr) into tree
			vptr->parent_in = min_aptr;
			if (uptr->first_child == NILN)
			{
				uptr->first_child = vptr;
				vptr->right_sibl = vptr;
				vptr->left_sibl = vptr;
			}
			else
			{
				vptr->right_sibl = uptr->first_child->right_sibl;
				uptr->first_child->right_sibl->left_sibl = vptr;
				vptr->left_sibl = uptr->first_child;
				uptr->first_child->right_sibl = vptr;
			}


			// subtree rooted at v has u as parent node now, update level and cost
			// entries of its nodes accordingly and produce list of nodes contained
			// in subtree
			upd_nodes = NILN;
			update_level = uptr->level + 1L;

			update_subtree(vptr);
			// now compute new keys of arcs into nodes that have acquired a new
			// shortest path, such arcs have head or tail in the subtree rooted at
			// "vptr", update vertex keys at the same time, nodes to be checked are
			// those contained in the subtree and the ones pointed to by arcs
			// emanating from nodes in the subtree
			vptr = upd_nodes;
			while (vptr != NILN)
			{
				if (vptr->vkey != NILA)
					DELETE_ROBUST(&h, vptr->vkey, epsilon_cost_time_ratio);
				min = DBL_MAX;
				vmin_aptr = NILA;
				aptr = vptr->first_arc_in;
				while (aptr != NILA)
				{
					if (!aptr->in_tree)
					{
						uptr = aptr->tail;
						if (uptr->transit_time_t + aptr->transit_time - vptr->transit_time_t > epsilon_transit_time) {
							aptr->key = (uptr->cost_t + aptr->cost - vptr->cost_t) / (uptr->transit_time_t + aptr->transit_time - vptr->transit_time_t);
						}
						else {
							aptr->key = infty;
						}

						if (min - aptr->key > epsilon_cost_time_ratio)
						{
							min = aptr->key;
							vmin_aptr = aptr;
						}
					}
					aptr = aptr->next_in;
				}
				if (vmin_aptr != NILA) {
					INSERT_ROBUST(&h, vmin_aptr, epsilon_cost_time_ratio);
				}
				vptr->vkey = vmin_aptr;
				vptr = vptr->link;
			}

			min_aptr->key = DBL_MAX;

			// now update keys of arcs from nodes in subtree to nodes not contained
			// in subtree and update vertex keys for the latter if necessary
			vptr = upd_nodes;
			while (vptr != NILN)
			{
				aptr = vptr->first_arc_out;
				while (aptr != NILA)
				{
					if (!aptr->in_tree && !aptr->head->in_list)
					{
						wptr = aptr->head;
						if (vptr->transit_time_t + aptr->transit_time - wptr->transit_time_t > epsilon_transit_time) {
							akey = (vptr->cost_t + aptr->cost - wptr->cost_t) / (vptr->transit_time_t + aptr->transit_time - wptr->transit_time_t);
						}
						else {
							akey = infty;
						}
						if (wptr->vkey->key - akey > epsilon_cost_time_ratio)
						{
							DELETE_ROBUST(&h, wptr->vkey, epsilon_cost_time_ratio);
							aptr->key = akey;
							INSERT_ROBUST(&h, aptr, epsilon_cost_time_ratio);
							wptr->vkey = aptr;
						}
						else {
							aptr->key = akey;
						}
					}
					aptr = aptr->next_out;
				}
				vptr = vptr->link;
			}

			vptr = upd_nodes;
			while (vptr != NILN)
			{
				vptr->in_list = false;
				vptr = vptr->link;
			}
		}

		DEALLOC_HEAP(&h);
		if (cycle != NULL && len != NULL)
		{
			*len = 0L;
			if (min_aptr != NILA)
			{
				cycle[(*len)++] = min_aptr;
				aptr = min_aptr->tail->parent_in;
				// MG: adapted the loop to work also for critical self-edges
				// keeping the original in comment for later reference in case of
				// problems or doubts...
				//    do
				//    {
				//        cycle[(*len)++] = aptr;
				//        aptr = aptr->tail->parent_in;
				//    }
				//    while (aptr->head != min_aptr->head);
				while (aptr->head != min_aptr->head)
				{
					cycle[(*len)++] = aptr;
					aptr = aptr->tail->parent_in;
				}
			}
		}
	}



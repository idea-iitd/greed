#define GUROBI
#include "src/env/ged_env.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

using NodeLabel = int;
//using EdgeLabel = ged::NoLabel;
using EdgeLabel = int;

class SEDEditCosts: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 0;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		return node_label_1 != node_label_2;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 0;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		return 0;
	}
};

class GEDEditCosts: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		return node_label_1 != node_label_2;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		return 0;
	}
};

ged::Options::GEDMethod method_name_to_option(std::string name)
{
	if (name == "anchor_aware_ged") {
		return ged::Options::GEDMethod::ANCHOR_AWARE_GED;
	} else if (name == "blp_no_edge_labels") {
		return ged::Options::GEDMethod::BLP_NO_EDGE_LABELS;
	} else if (name == "branch") {
		return ged::Options::GEDMethod::BRANCH;
	} else if (name == "f2") {
		return ged::Options::GEDMethod::F2;
	} else if (name == "ipfp") {
		return ged::Options::GEDMethod::IPFP;
	} else {
		throw std::invalid_argument("unknown method");
	}
}

using Data = std::pair< std::vector< NodeLabel >, std::vector< std::pair< int, int >>>;

std::tuple< double, double >
sed(const Data& g, const Data& h, std::vector< std::string > method_name, std::vector< std::string > method_args)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	
	auto gi = env.add_graph();
	const auto& g_x = g.first;
	const auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (const auto& p: g_edge_index) {
		//env.add_edge(gi, p.first, p.second, ged::NoLabel());
		env.add_edge(gi, p.first, p.second, 0);
	}
	
	auto hi = env.add_graph();
	const auto& h_x = h.first;
	const auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (const auto& p: h_edge_index) {
		//env.add_edge(hi, p.first, p.second, ged::NoLabel());
		env.add_edge(hi, p.first, p.second, 0);
	}
	
	// quick-fix: remove
	if (method_name[0] == "ged_f2") {
		env.set_edit_costs(new GEDEditCosts());
		method_name[0] = "f2";
	} else if (method_name[0] == "ged_branch") {
		env.set_edit_costs(new GEDEditCosts());
		method_name[0] = "branch";
	} else {
		env.set_edit_costs(new SEDEditCosts());
	}
	
	env.init();
	double lb, ub;
	if (method_name.size() == 1) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	} else if (method_name.size() == 2) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		env.set_method(method_name_to_option(method_name[1]), method_args[1]);
		env.init_method();
		env.run_method(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	}
	
	return std::make_tuple(lb, ub);
}

std::tuple< double, double, std::vector< int >, std::vector< int >>
sed_plus(const Data& g, const Data& h,
		std::vector< std::string > method_name, std::vector< std::string > method_args)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	
	auto gi = env.add_graph();
	const auto& g_x = g.first;
	const auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (const auto& p: g_edge_index) {
		//env.add_edge(gi, p.first, p.second, ged::NoLabel());
		env.add_edge(gi, p.first, p.second, 0);
	}
	
	auto hi = env.add_graph();
	const auto& h_x = h.first;
	const auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (const auto& p: h_edge_index) {
		//env.add_edge(hi, p.first, p.second, ged::NoLabel());
		env.add_edge(hi, p.first, p.second, 0);
	}
	
	env.set_edit_costs(new SEDEditCosts());
	env.init();
	double lb, ub;
	if (method_name.size() == 1) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	} else if (method_name.size() == 2) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		env.set_method(method_name_to_option(method_name[1]), method_args[1]);
		env.init_method();
		env.run_method(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	}
	
	ged::NodeMap node_map = env.get_node_map(gi, hi);
	
	std::vector< int > node_mask_idx;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		if (node_map.pre_image(i) != ged::GEDGraph::dummy_node()) {
			node_mask_idx.push_back(i);
		}
	}
	
	std::vector< std::vector< bool >> g_adj(g_x.size(), std::vector< bool >(g_x.size(), false));
	for (const auto &p: g_edge_index) {
		g_adj[p.first][p.second] = true;
	}
	
	std::vector< int > edge_mask_idx;
	for (int i = 0; i < (int)h_edge_index.size(); ++i) {
		const auto &p = h_edge_index[i];
		if (
				(node_map.pre_image(p.first) != ged::GEDGraph::dummy_node()) &&
				(node_map.pre_image(p.second) != ged::GEDGraph::dummy_node()) &&
				g_adj[node_map.pre_image(p.first)][node_map.pre_image(p.second)]
			) {
			edge_mask_idx.push_back(i);
		}
	}
	
// 	std::vector< std::pair< std::size_t, std::size_t >> rel;
// 	node_map.as_relation(rel);
// 	std::cout << std::endl;
// 	for (const auto &p: rel) {
// 		std::cout << p.first << " -> " << p.second << std::endl;
// 	}
// 	std::cout << std::endl;
	
	return std::make_tuple(lb, ub, node_mask_idx, edge_mask_idx);
}

std::pair< std::vector< std::pair< std::size_t, std::size_t >>, double >
sed_align(const Data& g, const Data& h)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	auto gi = env.add_graph();
	auto& g_x = g.first;
	auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (auto& p: g_edge_index) {
		//env.add_edge(gi, p.first, p.second, ged::NoLabel());
		env.add_edge(gi, p.first, p.second, 0);
	}
	auto hi = env.add_graph();
	auto& h_x = h.first;
	auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (auto& p: h_edge_index) {
		//env.add_edge(hi, p.first, p.second, ged::NoLabel());
		env.add_edge(hi, p.first, p.second, 0);
	}
	env.set_edit_costs(new SEDEditCosts());
	env.init();
	env.set_method(ged::Options::GEDMethod::BRANCH);
	env.init_method();
	env.run_method(gi, hi);
	ged::NodeMap node_map = env.get_node_map(gi, hi);
	std::vector< std::pair< std::size_t, std::size_t >> rel;
	node_map.as_relation(rel);
	return std::make_pair(rel, node_map.induced_cost());
}

PYBIND11_MODULE(pyged, m) {
	m.def("sed", &sed);
	m.def("sed_plus", &sed_plus);
	m.def("sed_align", &sed_align);
}


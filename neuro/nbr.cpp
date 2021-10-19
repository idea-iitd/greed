/**
 * optimized neighborhood decomposition
 * @author: Rishabh Ranjan
 */

#include <bits/stdc++.h>
using namespace std;

#define rep(i, n) for (int i = 0; i < n; ++i)

const int N = 3177888;
const int M = 8466859;
const int K = 2;
const int C = 8;
const int L = 1000;
const int T = 10000;

int id_to_label[N];
vector<int> adj[N];
bool seen[N];
int local_id[N];
bool global_seen[N];
unordered_set<int> global_nbr[N];

int main()
{
    cout << "expected nodes: " << N << endl;
    cout << "expected edges: " << M << endl;
    cout << "number of hops: " << K << endl;
    cout << "reduced labels: " << C << endl;
    cout << "nbr node limit: " << L << endl;
    cout << "print interval: " << T << endl;
    cout << endl;

    cout << "reading labels.txt..." << endl;
    ifstream lst("labels.txt");
    rep(i, N) {
        if (i % T == 0) cout << i << "\r";
        int tmp; lst >> tmp >> tmp;
        id_to_label[i] = tmp % C;
    }
    cout << endl;
    lst.close();

    cout << "reading edges.txt..." << endl;
    ifstream est("edges.txt");
    rep(i, M) {
        if (i % T == 0) cout << i << "\r";
        int u, v; est >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    cout << endl;
    est.close();

    cout << "decomposing..." << endl;
    int n_nbrs = 0; double n_nodes = 0, n_edges = 0;
    ofstream nst("nbrs.txt");
    rep(src, N) {
        if (src % T == 0) {
            cout << src << "\r"; cout.flush();
        }
        if ((int)adj[src].size() == 0) continue;
        vector<int> from, to;
        memset(seen, 0, sizeof seen);
        vector<int> ftr({src}); seen[src] = true;
        global_seen[src] = true;
        bool flag = false; ostringstream tst;
        local_id[src] = 0; tst << id_to_label[src] << ' ';
        int cur_id = 1;
        rep(hop, K) {
            vector<int> new_ftr;
            for (int node: ftr) {
                for (int nbr: adj[node]) {
                    if (!seen[nbr]) {
                        if (cur_id >= L) {
                            flag = true;
                            break;
                        }
                        new_ftr.push_back(nbr); seen[nbr] = true;
                        global_seen[nbr] = true;
                        local_id[nbr] = cur_id++; tst << id_to_label[nbr] << ' ';
                    }
                    from.push_back(node); to.push_back(nbr);
                    global_nbr[node].insert(nbr);
                }
                if (flag) break;
            }
            if (flag) break;
            swap(ftr, new_ftr);
        }
        if (flag) continue; nst << tst.str();
        for (int node: ftr) {
            for (int nbr: adj[node]) if (seen[nbr]) {
                from.push_back(node); to.push_back(nbr);
                global_nbr[node].insert(nbr);
            }
        }
        ++n_nbrs; n_nodes += cur_id; n_edges += from.size()/2;
        nst << '\n';
        for (node: from) nst << local_id[node] << ' ';
        nst << '\n';
        for (node: to) nst << local_id[node] << ' ';
        nst << '\n';
    }

    cout << endl;
    nst.close();
    cout << endl;
    cout << "number of nbrs: " << n_nbrs << endl;
    cout << "averaged nodes: " << n_nodes/n_nbrs << endl;
    cout << "averaged edges: " << n_edges/n_nbrs << endl;
    int n_induced_nodes = 0; rep(node, N) n_induced_nodes += global_seen[node];
    cout << "induced nodes: " << n_induced_nodes << endl;
    int n_induced_edges = 0; rep(node, N) n_induced_edges += global_nbr[node].size();
    cout << "induced edges: " << n_induced_edges/2 << endl;
}

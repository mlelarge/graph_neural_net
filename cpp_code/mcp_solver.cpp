#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace std;

template <typename T>
void print_it(T iterable, bool end_line=false){
    for (auto const &e: iterable){
        cout << e << ' ';
    }
    if (end_line){cout << endl;}
}

template <typename T>
void print_matrix(vector<vector<T>> adj){
    for (unsigned int i=0; i!=adj.size();i++){
        for (unsigned int j=0; j!=(adj[i]).size(); j++){
            cout << (adj[i][j]) << ' ';
        }
        cout << endl;
    }
}

template <typename T>
set<T> My_union(set<T> A, set<T> B){
    set<T> AuB(A);
    AuB.insert(B.begin(),B.end());
    return AuB;
}

template <typename T>
set<T> My_intersection(set<T> A, set<T> B){
    set<T> AnB;
    set_intersection(A.begin(),A.end(),B.begin(),B.end(),inserter(AnB,AnB.begin()));
    return AnB;
}

template <typename T>
set<T> My_diff(set<T> A, set<T> B){
    set<T> AminusB;
    set_difference(A.begin(),A.end(),B.begin(),B.end(),inserter(AminusB,AminusB.begin()));
    return AminusB;
}

template <typename T>
T Sample_elt(set<T> A){
    int length = A.size();
    int random = rand() % length;
    T elt;
    int counter = 0;
    typename set<T>::iterator it;
    for (it=A.begin();it!=A.end();it++){
        if (counter==random){
            elt = *it;
        }
        counter++;
    }
    return elt;
}

set<int> Fast_copy(set<int> set_to_copy){
    set<int> copied_set;
    copy(set_to_copy.begin(),set_to_copy.end(),inserter(copied_set,copied_set.begin()));
    return copied_set;
}

set<int> Neighs(int u, const vector<vector<bool>>* adj){
    set<int> neighbours;
    const vector<bool> possibles = adj->at(u);
    for (int i=0;i!=possibles.size();++i){
        if (possibles[i]){
            neighbours.insert(i);
        }
    }
    return neighbours;
}

void _bronk2(vector<set<int>> *solutions, set<int> R, set<int> P, set<int> X, const vector<vector<bool>>* adj){
    if (P.empty() && X.empty()){
        set<int> added_cycle;
        copy(R.begin(),R.end(),inserter(added_cycle, added_cycle.begin()));
        solutions->push_back(added_cycle);
    }else{
        set<int> PuX = My_union(P,X);
        int u = Sample_elt(PuX);
        set<int> N_u = Neighs(u,adj);
        set<int>N_v;
        set<int> PminusN_u = My_diff(P,N_u);
        set<int>::iterator it;
        int v;
        set<int> newR;
        set<int> newP = Fast_copy(P);
        set <int>newX = Fast_copy(X);
        for (it=PminusN_u.begin();it!=PminusN_u.end();++it){
            v = *it;
            N_v = Neighs(v,adj);
            newR = Fast_copy(R);
            newR.insert(v);
            _bronk2(solutions, newR , My_intersection(newP,N_v), My_intersection(newX,N_v),adj);
            newP.erase(v);
            newX.insert(v);
        }

    }

}

vector<set<int>> Bronk2(vector<vector<bool>> adj){
    int n = adj.size();
    set<int> base_set;
    for (int i=0;i!=n;i++){
        adj[i][i]=false;
        base_set.insert(i);
    }
    vector<set<int>> cliques;
    _bronk2(&cliques, set<int>(),base_set,set<int>(),&adj);

    set<int>::iterator it;
    int max_length=0;
    int cur_length;
    set<int> cur_clique;
    vector<set<int>> max_cliques;
    for (int i=0;i!=cliques.size();++i){
        cur_clique = cliques[i];
        cur_length = cur_clique.size();
        if (cur_length==max_length){
            max_cliques.push_back(cur_clique);
        }else if (cur_length>max_length){
            max_cliques.clear();
            max_cliques.push_back(cur_clique);
            max_length = cur_length;
        }
    }
    return max_cliques;
}

vector<vector<bool>> read_data(string filename){
    vector<vector<bool>> adj;
    ifstream my_file(filename);
    if (my_file.is_open()){
        string line;
        vector<bool> cur_row;
        bool value;
        while (getline(my_file,line)){
            for(unsigned int i=0;i!=line.size();i++){
                if ((line[i])!=' '){
                    value = (line[i]=='1');
                    cur_row.push_back(value);
                }
            }
            adj.push_back(cur_row);
            cur_row.clear();
        }
        my_file.close();
    }else{
        cout << "Couldn't read file '" << filename << "' !";
    }

    return adj;
}

void write_data(string filename, vector<set<int>> cliques){
    int n_cliques = cliques.size();
    int cs = ((cliques[0]).size());
    string cur_clique = "";
    ofstream my_file(filename);
    if (my_file.is_open()){
        for (unsigned int i=0; i!=n_cliques; i++){
            for (const auto &e: (cliques[i])){
                cur_clique += to_string(e);
                cur_clique.push_back(' ');
            }
            cur_clique.pop_back();
            my_file << cur_clique;
            cur_clique = "";
            if (i!=n_cliques-1){
                my_file << "\n";
            }
        }
    }else{
        cout << "Could not open file '" << filename << "' !";
    }
}

void test(){
    std::set<int> first = { 1, 3, 5, 6, 8 };
    std::set<int> second = { 2, 4, 6 };
    set<int> sec_copy = Fast_copy(second);
    second.insert(8);
    std::cout << "Testing copy : " << endl << "Copy : ";
    for (auto const &e: sec_copy){cout << e << ' ';}
    cout << endl << "Original updated : ";
    for (auto const &e: second){cout << e << ' ';}

    cout << endl << endl;


    vector<vector<bool>> adj = { {false, true, true, true},
                                 {true, false, false, true},
                                 {true, true, true, true},
                                 {false, false, false, false}};
    set<int> neighbours = Neighs(0,&adj);
    cout << "Testing Neighs : ";
    for (auto const &e: neighbours) {
        std::cout << e << ' ';
    }
    cout << endl << endl;

    cout << "Testing intersection : ";
    std::set<int> s;
    set_intersection(first.begin(),first.end(),second.begin(),second.end(),inserter(s,s.begin()));


    for (auto const &e: s) {
        std::cout << e << ' ';
    }
    cout << endl << endl;

    cout << "Testing sampling : " <<Sample_elt(first) << endl << endl;

    adj = {{false, true , false, false, true , false},
                                {true , false, true, false, true , false},
                                {false, true , false, true , false, false},
                                {false, false, true , false, true , true },
                                {true , true , false, true , false, false},
                                {false, false, false, true , false, false}};

    vector<set<int>> cliques = Bronk2(adj);
    cout << "Testing Bronk2 : output should be '0 1 4' :" << endl;
    for(int i=0; i!=cliques.size();i++){
        print_it(cliques[i]);
        cout << endl;
    }

    cout << endl << "Testing reading + Bronk2" << endl;
    adj = read_data("input_test.txt");
    print_matrix(adj);
    cliques = Bronk2(adj);
    for(int i=0; i!=cliques.size();i++){
        print_it(cliques[i]);
        cout << endl;
    }
    cout << endl << "Testing writing in 'output_test.txt'" << endl;
    write_data("output_test.txt", cliques);
}

void Compute(string filename, bool verbose){
    vector<vector<bool>> adj = read_data(filename);
    if (adj.empty()){
        cout << endl << "No data read. Skipping " << filename <<endl;
    }else{
        if (verbose){cout << "Data read" << endl;}
        vector<set<int>> cliques = Bronk2(adj);
        if (verbose){cout << "Solution found" << endl;}
        string out_name(filename);
        out_name+="s";
        write_data(out_name,cliques);
        if (verbose){cout << "Solution written in " << out_name<< endl;}
    }
}


int main(int argc, char* argv[]){

    string verbose_str = "-v";
    bool verbose = false;
    vector<string> filenames;
    if (argc==1){
        //cout << "Computing for file : " << "input_test.txt" << endl;
        Compute("input_test.txt",false);
    }else{
        for (unsigned int i=1; i!=argc;i++){
            if ((argv[i])==verbose_str){
                verbose=true;
            }else{
                filenames.push_back((argv[i]));
            }
        }
        for (vector<string>::iterator fname=filenames.begin(); fname!=filenames.end();++fname){
            if (verbose){cout << "Solving " << *fname << endl;}
            Compute(*fname,verbose);
        }
    }
    return 0;
}

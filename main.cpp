#include "boost.h"

using namespace std;

int main(void)
{
     double scores = 0.0;
	double summ = 0.0;
	CART_binar_classifier cart;
	vector<vector<double> >* A = new vector<vector<double> >;
	vector<vector<double> >* C = new vector<vector<double> >;
	vector<vector<double> >* L = new vector<vector<double> >;

	vector<int>* B = new vector<int>;
	vector<int>* D = new vector<int>;

	vector<int>* E = new vector<int>;
	vector<int>* P = new vector<int>;

	vector<double>* G = new vector<double>;

	vector<double> temp;

	char tr_name[] = "/data/train.data";
	char tr_answ[] = "/data/train.answ";

	read_data_from_file(tr_name,tr_answ,A,B);

	cross_validation(A,B,C,D,L,P,0.66);
	
	cart.train(C,D);
	cart.predict(L,E);

	score(E,P); 

	binary_cart_boost_classifier boost(200,0.5,0.8);
	boost.train(C,D);
	boost.predict(L,G);

	for(int i = 0;i < boost.number_of_models;i++)
		summ += boost.weights[i];
	summ /= boost.number_of_models;

	E->clear();

	for(int i = 0;i < G->size();i++)
	{
		if((*G)[i] >= summ)
			E->push_back(1);

		else E->push_back(0);
	}
	
	score(E,P);
	
     A->clear();delete A;
     B->clear();delete B;
	C->clear();delete C;
	D->clear();delete D;
	E->clear();delete E;
	G->clear();delete G;
	P->clear();delete P;
	L->clear();delete L;
	
	return 0;
}

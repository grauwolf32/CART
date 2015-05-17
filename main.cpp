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
	vector<double>* As = new vector<double>;
	vector<int>* E = new vector<int>;
	vector<int>* P = new vector<int>;
	

	vector<double> temp;

	char tr_name[] = "train.data";
	char tr_answ[] = "train.answ";

	read_data_from_file(tr_name,tr_answ,A,B);
	cross_validation(A,B,C,D,L,P,0.667);
	
	cart.train(C,D);
	
	binary_cart_boost_classifier boost(100,0.5,0.5);

	int num = 10;
	double min_score = 10e+6;
	double score_tmp = 0.0;
	double cart_min = 10e+6;
	double cart_tmp = 0.0;

	//cross_validation(A,B,C,D,L,P,0.667);
	boost.train(C,D);

	for(int i = 0;i < num;i++)
	{	
		cross_validation(A,B,C,D,L,P,0.667);
		
		boost.predict(L,As);

		E->clear();
		cart.predict(L,E);
		cart_tmp = score(E,P); 
		if(cart_min > cart_tmp)
			cart_min = cart_tmp;
		
	
		for(int i = 0;i < boost.number_of_models;i++)
			summ += boost.weights[i];
		summ /= boost.number_of_models;

		E->clear();
		
		for(int i = 0;i < As->size();i++)
		{
			if((*As)[i] >= summ)
				E->push_back(1);

			else E->push_back(0);
		}
		
		score_tmp = score(E,P);
		if(min_score > score_tmp)
			min_score = score_tmp;
		
	}
	for(int i = 0;i < boost.number_of_models;i++)
		cout << " "<<boost.weights[i]<<" ";
	cout << "\n";
	cout <<"Score cart: "<<cart_min<<"\n";
	cout <<"Score boost: "<<min_score<<"\n";

	//cart.print_tree();
     A->clear();delete A;
     B->clear();delete B;
	C->clear();delete C;
	D->clear();delete D;
	E->clear();delete E;
	
	return 0;
}

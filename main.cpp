#include "boost.h"

using namespace std;

int main(void)
{
     double scores = 0.0;
	double summ = 0.0;
	CART_binar_classifier cart;
	vector<vector<double> >* A = new vector<vector<double> >;
	vector<vector<double> >* C = new vector<vector<double> >;

	vector<int>* B = new vector<int>;
	vector<int>* D = new vector<int>;
	vector<double>* As = new vector<double>;
	vector<int>* E = new vector<int>;

	vector<double> temp;

	char tr_name[] = "train.data";
	char tr_answ[] = "train.answ";

	char ts_name[] = "test.data";
	char ts_answ[] = "test.answ";

	read_data_from_file(tr_name,tr_answ,A,B);
	read_data_from_file(ts_name,ts_answ,C,D);
	
	cart.train(A,B);
	cart.predict(A,E);

	scores = score(E,B); 
	cout <<"score (C): "<< scores << "\n";

	/*
	//cart.print_tree();
	
	binary_cart_boost_classifier boost(30,0.6,0.6);

	boost.train(A,B); 
	boost.predict(C,As);

	for(int i = 0;i < boost.number_of_models;i++)
		summ += boost.weights[i];
	summ /= boost.number_of_models;

	E->clear();

	cout <<"As size: "<<As->size()<<" C size: "<< C->size()<<" D size :"<<D->size();
	for(int i = 0;i < As->size();i++)
	{
		if((*As)[i] >= summ)
			E->push_back(1);

		else E->push_back(0);
	}
	/*
	for(int i = 0;i < As->size();i++)
	{
		if(is_equ((*As)[i],1.0))
			E->push_back(1);

		else E->push_back(0);
	}
	*/
	/*
	for(int i = 0; i < boost.number_of_models;i++)
	{
		cout << "Weight "<<i<<" "<<boost.weights[i]<<"\n";
	}
	
	scores = score(D,E); 
	cout <<"score (C): "<< scores << "\n";

	As->clear();
	boost.predict(A,As);

     summ = 0.0;
	for(int i = 0;i < boost.number_of_models;i++)
		summ += boost.weights[i];
	summ /= boost.number_of_models;

	E->clear();
	for(int i = 0;i < As->size();i++)
	{
		if(fabs((*As)[i] >= summ))
			E->push_back(1);

		else E->push_back(0);
	}
	scores = score(B,E); 
	cout <<"score (A): "<< scores << "\n";
	/*
	vector<vector<double> >* S_D = new vector<vector<double> >;
	vector<int>* S_A = new vector<int>;

	vector<vector<double> >* S_TD = new vector<vector<double> >;
	vector<int>* S_TA = new vector<int>;

     vector<int>* selected_data = new vector<int>;
     vector<int>* selected_features = new vector<int>;
	*/
	/*
	rsm(A,B,S_D,S_A,0.6,0.6,selected_data,selected_features);
    
	
	for(int i = 0;i < S_D->size();i++)
	{
		for(int j = 0;j < (*S_D)[i].size();j++)
		{
			cout <<(*S_D)[i][j]<<" ";
		}
          cout << "Ans: "<<(*S_A)[i]<<" \n";
	}
	
	
	double n_prc = 0.0;
	for(int i = 0; i < (int)S_A->size();i++)
	{
		if((*S_A)[i] == 1)n_prc += 1.0;
	}
	n_prc /= (int)S_A->size();
	cout <<"Percent: "<<n_prc<<"\n";

     cart.train(S_D,S_A);
	sub_space(C,S_TD,selected_features);
	cart.predict(S_TD,E);

	scores = score(D,E); 
	cout <<"score: "<< scores << "\n";

	selected_data->clear();delete selected_data;
	selected_features->clear();delete selected_features;

	S_A->clear();delete S_A;
	S_D->clear();delete S_D;

	S_TA->clear();delete S_TA;
	S_TD->clear();delete S_TD;
	*/
	
     A->clear();delete A;
     B->clear();delete B;
	C->clear();delete C;
	D->clear();delete D;
	E->clear();delete E;
	
	return 0;
}

#include "boost.h"

using namespace std;

binary_cart_boost_classifier::binary_cart_boost_classifier(int number_of_mod,double data_prc_,double feature_prc_)
{
	number_of_models = number_of_mod;
	weights = new double[number_of_models];

	for(int i = 0;i < number_of_models;i++)
	{
		weights[i] = 0.0;
	}

	models = new CART_binar_classifier[number_of_models];
	data_prc = data_prc_;
	feature_prc = feature_prc_;
}

binary_cart_boost_classifier::~binary_cart_boost_classifier()
{
	if(weights != NULL)delete[] weights;
	if(models != NULL) delete[] models;
}

void binary_cart_boost_classifier::predict(vector<vector<double> >* data_in,vector<double>* predicted)
{
	predicted->clear();
	int n = data_in->size();

	vector<vector<double> > data_i;
	vector<int>  predicted_i;

	for(int i = 0;i < n;i++)
		predicted->push_back(0.0);
	
	for(int i = 0;i < number_of_models;i++)
	{
		if(is_equ(weights[i],0.0))
		{
			continue;
		}

		sub_space(data_in,&data_i,selected_features[i]);
		models[i].predict(&data_i,&predicted_i);

		for(int j = 0;j < n;j++)
		{
			(*predicted)[j] += weights[i]*predicted_i[j];
		}

	}
}

/*
void binary_cart_boost_classifier::predict(vector<vector<double> >* data_in,vector<double>* predicted)
{
	predicted->clear();
	int n = data_in->size();

	vector<vector<double> > data_i;
	vector<int>  predicted_i;
	vector<pair<double,double> > vote_pred(n);

	for(int i = 0;i < n;i++)
	{
		predicted->push_back(0.0);
		vote_pred[i].first  = 0.0;
		vote_pred[i].second = 0.0;
	}
	
	for(int i = 0;i < number_of_models;i++)
	{
		if(is_equ(weights[i],0.0))
		{
			continue;
		}

		sub_space(data_in,&data_i,selected_features[i]);
		models[i].predict(&data_i,&predicted_i);

		for(int j = 0;j < n;j++)
		{
			if(predicted_i[j] == 0)
				vote_pred[j].first += weights[i];

			if(predicted_i[j] == 1)
				vote_pred[j].second += weights[i];
		}
	}

	for(int j = 0;j < n;j++)
	{
		if(vote_pred[j].first < vote_pred[j].second)
			(*predicted)[j] = 1.0;
	}

}

*/
void binary_cart_boost_classifier::train(vector<vector<double> >* data_,vector<int>* answ_,int period)
{
	if(period > max_period)
		return;

	vector<vector<double> > G;
	vector<double>    h;
	vector<double> temp;

	double g_ij = 0.0;
	double b_i = 0.0;
	double w_i = 0.0;
	double beta = 0.0;
	double a_sum = 0.0;

	double min_square_err = -1.0;
	double square_err = 0.0;
	double L_1_err = 0.0;
	double L_err = 0.0;

	int index = 0;
	int y_j = 0;
	int a  =  0;
	

	if(period == 0)
	{
		data = data_;
		answ = answ_;

		for(int i = 0;i < number_of_models;i++)
		{
			vector<vector<double> >* data_out = new vector<vector<double> >;
			vector<int>* 			answ_out = new vector<int>;

			vector<int>* 			selected_data_ = new vector<int>;
			vector<int>*			selected_feature_ = new vector<int>;
			
			rsm(data,answ,data_out,answ_out,data_prc,feature_prc,selected_data_,selected_feature_);

			selected_data.push_back(selected_data_);
			selected_features.push_back(selected_feature_);

			models[i].train(data_out,answ_out);
		}
		double min_err = 0.0;
		int min_i = 0;

		for(int i = 0;i < number_of_models;i++)
		{
			weights[i] = 1.0;
			predict(data,&h);
			
			L_err = L(h,(*answ));

			if(is_equ(min_err,0.0))
			{
				min_err = L_err;
				min_i = i;
			}

			if(L_err < min_err)
			{
				min_err = L_err;
				min_i = i;
			}
			weights[i] = 0.0;
		}	

		
		weights[min_i] = 1.0;
		cout <<"Start error: "<< min_err <<"\n";
		
		
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0,0.5);
		
		for(int i = 0;i < number_of_models;i++)
		{
			weights[i] += dis(gen);
		}
		

	}

	for(int i = 0; i < number_of_models;i++)
	{
		temp.clear();
		predict(data,&h);

		for(int j = 0; j < (int)h.size();j++)
		{
			y_j = (*answ)[j];
			temp.push_back(y_j - sigm(h[j]));
		}
		G.push_back(temp);
	}
		
	for(int i = 0; i < number_of_models;i++)
	{
		vector<vector<double> > data_i;
		vector<double> a_i;

		sub_space(data,&data_i,selected_features[i]);
		int k = data_i.size();

		beta = 0.0;
		a_sum = 0.0;
		square_err = 0.0;

		for(int j = 0;j < k;j++)
		{
			a = models[i].predict(data_i[j]);
			a_i.push_back(a);
			beta += G[i][j]*a;
			a_sum += a*a;
		}
		beta /= a_sum;

		for(int j = 0;j < k;j++)
		{
			square_err += (G[i][j] - beta*a_i[j])*(G[i][j] - beta*a_i[j]);
		}

		if(min_square_err < 0.0)
		{
			min_square_err = square_err;
			index = i;
		}

		if(square_err <= min_square_err)
		{
			min_square_err = square_err;
			index = i;
		}
	}
	
	b_i = coeff(period);
	w_i = weights[index];	

	/*
	predict(data,&h);
	L_1_err = L(h,(*answ));
	*/

	weights[index] += sign(beta)*b_i;

	predict(data,&h);
	L_err = L(h,(*answ));
	cout <<"L("<<period<<") :"<<L_err<<"\n"; 

	/*if(L_1_err < L_err )return;*/
	
	period += 1;
	train(data_,answ_,period);
}

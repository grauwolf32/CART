#include "utilites.h"

using namespace std;

double gini_impurity(double n,double n1)
{	
	if(n <= 0)
		return -1.0;

	double res = 1.0 - (n1/n)*(n1/n) - ((n-n1)/n)*((n-n1)/n);
	return res;
}

double d_imp(int n,int n_l, int n_1, int n_1_l) 
{
	double n_r = n - n_l;
	double n_2_l = n_l - n_1_l;
	double n_1_r = n_1 - n_1_l;
	double n_2_r = n_r - n_1_r;

	if(n_l <= 0.001 || n_r <= 0.001)
		  return 0.0;	
	
	return (1.0/(double)n_l)*(double)( (n_1_l)*(n_1_l) + (n_2_l)*(n_2_l) )\
		+ (1.0/(double)n_r)*(double)( (n_1_r)*(n_1_r) + (n_2_r)*(n_2_r) );  
}	

double coeff(int period)
{
	return min_step - (double)period/max_period;
}

double sign(double x)
{
	if(x > 0.000001)return 1.0;
	if(x < -0.00001)return -1.0;
	return 0.0;
}

double L(vector<double>& h,vector<int>& y)
{
	int n = (int)h.size();
	double ans = 0.0;
	for(int i = 0;i < n;i++)
	{
		ans += (-1.0*y[i]*log(sigm(h[i])) - (1.0-y[i])*log(1.0 - sigm(h[i])));
	}
	return ans;
}

void read_data_from_file(char* data,char* answ,vector<vector<double> >* datas,vector<int>* answers)
{
	int m = 0;
	int n = 0;

	int k = 0;
	double temp  = 0.0;	
	vector<double> tmp;

	ifstream data_file;
	ifstream answ_file;

	answ_file.open(answ);
	data_file.open(data);

	data_file >> m >> n;

	for(int i = 0;i < m;i++)
	{
		for(int j = 0;j < n;j++)
		{
			data_file >> temp;
			tmp.push_back(temp);
		}
		datas->push_back(tmp);
		tmp.clear();
	}
	
	for(int i = 0;i < m;i++)
	{
		answ_file >> temp;
		if(is_equ(temp,0.0))
			k = 0;

		else k = 1;
		
		answers->push_back(k);
	}

	data_file.close();
	answ_file.close();
}

void sub_space(vector<vector<double> >* data_in,vector<vector<double> >* data_out,vector<int>* selected_features)
{
	data_out->clear();
	
	int k = data_in->size();
	int l = selected_features->size();

	vector<double> temp;
	
	for(int i = 0;i < k;i++)
	{
		temp.clear();
		for(int j = 0;j < l;j++)
		{
			temp.push_back((*data_in)[i][(*selected_features)[j]]);
		}
		data_out->push_back(temp);
	}
	
}

void rsm(vector<vector<double> >* data_in,vector<int>* answ_in,\
	    vector<vector<double> >* data_out,vector<int>* answ_out,\
        double data_prc,double feature_prc,vector<int>* selected_data,\
								    vector<int>* selected_features)
{
	int m = (int)data_in->size();
	int n = (int)(*data_in)[0].size();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0,1);
	
	for(int i = 0;i < m;i++)
	{
		if(dis(gen) <= data_prc)
			selected_data->push_back(i); 
	}

	for(int i = 0;i < n;i++)
	{
		if(dis(gen) <= feature_prc)
			selected_features->push_back(i); 
	}

	data_out->clear();
	answ_out->clear();
	
	int k = selected_data->size();
	int l = selected_features->size();

	vector<double> temp;
	
	for(int i = 0;i < k;i++)
	{
		temp.clear();
		for(int j = 0;j < l;j++)
		{
			temp.push_back((*data_in)[(*selected_data)[i]][(*selected_features)[j]]);
		}
		data_out->push_back(temp);
		answ_out->push_back((*answ_in)[(*selected_data)[i]]);
	}
}

void cross_validation(vector<vector<double> >* data_in,vector<int>* answ_in,\
	    vector<vector<double> >* data_out_train,vector<int>* answ_out_train,\
	    vector<vector<double> >* data_out_test,vector<int>* answ_out_test,double data_prc)
{
	int m = (int)data_in->size();
	int n = (int)(*data_in)[0].size();

	data_out_train->clear();
	answ_out_train->clear();

	data_out_test->clear();
	answ_out_test->clear();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0,1);
	
	for(int i = 0;i < m;i++)
	{
		if(dis(gen) <= data_prc)
		{
			data_out_train->push_back((*data_in)[i]);
			answ_out_train->push_back((*answ_in)[i]);
		}
		else{
			data_out_test->push_back((*data_in)[i]);
			answ_out_test->push_back((*answ_in)[i]);
		}
	}
	
}

double sigm(double x)
{
	return 1.0/(1.0 + exp(-x));
}

int is_equ(double a,double b)
{
	if((a > b - eps) && (a < b + eps))return 1;
	return 0;
}

double score(vector<int>* predicted,vector<int>* answ)
{
	double scr = 0.0;
	double scr_0 = 0.0;
	double scr_1 = 0.0;

	int n = predicted->size();
	int k = 0;
	int l = 0;

	for(int i = 0;i < n;i++)
	{
		if((*predicted)[i] == (*answ)[i]){scr += 1.0;}
		if((*predicted)[i] == (*answ)[i] && (*answ)[i] == 0)
			scr_0 += 1.0;
		if((*predicted)[i] == (*answ)[i] && (*answ)[i] == 1)
			scr_1 += 1.0;

		if((*answ)[i] == 0){k++;}
		else {l++;}
	}

	scr /= n;
	scr_0 /= k;
	scr_1 /= l;

	cout <<"score: "<< scr << "\n";
	cout <<"score 0: "<< scr_0 << "\n";
	cout <<"score 1: "<< scr_1 << "\n";
	

	return scr;
}


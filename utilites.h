#ifndef __UTILITES_H__
#define __UTILITES_H__

#include <cmath>
#include <set>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <utility>
#include <vector>


using namespace std;

double gini_impurity(double n,double n1);
double d_imp(int n,int n_l, int n_1, int n_1_l);
double score(vector<int>* predicted,vector<int>* answ);
double L(vector<double>& h,vector<int>& y);
int is_equ(double a,double b);

double coeff(int period);
double sigm(double x);
double sign(double x);

void read_data_from_file(char* data,char* answ,vector<vector<double> >* datas,vector<int>* answers);
void sub_space(vector<vector<double> >* data_in,vector<vector<double> >* data_out,vector<int>* selected_features);

void rsm(vector<vector<double> >* data_in,vector<int>* answ_in,\
	    vector<vector<double> >* data_out,vector<int>* answ_out,\
        double data_prc,double feature_prc,vector<int>* selected_data,\
								    vector<int>* selected_features);

void cross_validation(vector<vector<double> >* data_in,vector<int>* answ_in,\
	    vector<vector<double> >* data_out_train,vector<int>* answ_out_train,\
	    vector<vector<double> >* data_out_test,vector<int>* answ_out_test,double data_prc);

int const max_period = 300;
double const min_step = 1.0;
double const eps = 0.000001;

typedef vector<pair<int,vector<double> > > train_data;
typedef vector<pair<double,int> > 		      feature;
#endif

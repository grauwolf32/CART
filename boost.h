#ifndef __BOOST_H__
#define __BOOST_H__

#include "utilites.h"
#include "cart.h"

using namespace std;

class binary_cart_boost_classifier
{
	public:
			binary_cart_boost_classifier(int number_of_models,double data_prc,double feature_prc);
			~binary_cart_boost_classifier();

			double 	predict(vector<double>& sample);
           	void predict(vector<vector<double> >* data,vector<double>* predicted);
			void train(vector<vector<double> >* data,vector<int>* answ_,int period = 0);
		
			CART_binar_classifier* models;
			double* weights;

			int number_of_models;
			double data_prc;
			double feature_prc;
	private:
			vector<vector<int>* > selected_features;
			vector<vector<int>* > selected_data;

			vector<vector<double> >* data;
			vector<int>* answ;
};

class boost_classifier
{
	public:
			boost_classifier(int num_trees,double data_prc,double feature_prc);
		    ~boost_classifier();
			void predict(vector<vector<double> >* data_in,vector<int>* answ_out);
			void train(vector<vector<double> >* data_,vector<int>* answ_);
			
	private:

			vector<vector<double> > data;
			vector<int> answ;
};

#endif

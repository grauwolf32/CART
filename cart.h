#ifndef __CART_H__
#define __CART_H__

#include "utilites.h"
#include "leaf.h"

using namespace std;

class CART_binar_classifier
{
	public:
			CART_binar_classifier();
			~CART_binar_classifier();

			void train(vector<vector<double> >* data,vector<int>* answ_);
               void print_tree();

			int 	predict(vector<double>& sample);
               void predict(vector<vector<double> >* data,vector<int>* predicted);
			Leaf* main_node;
	private:
			int m,n;
			void clear();
			void destroy_tree(Leaf* pCurrent);
			vector<vector<double> >* data_;
			vector<int>* 			answ_;
			train_data 			data;
	
			
};

#endif

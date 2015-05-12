#ifndef __LEAF_H__
#define __LEAF_H__
#include "utilites.h"

using namespace std;

class Leaf
{
	public:
			Leaf();
			Leaf(train_data* data,vector<int>* items,int n_1);
               ~Leaf();

			Leaf* pLeft;    
			Leaf* pRight;   
	
			double PrL,PrR; 
			double impurity;
			
			int classify(vector<double>& sample);
			void split();

			void print_leaf(int leaf_level = 0);
			void print_tree(int leaf_level = 0);

	private:	
			pair<double,int> s;
			vector<int>*  items;
			train_data*   data;
		     int  label; 
			int    n_1; 			
};

#endif

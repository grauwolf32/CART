#ifndef __LEAF_H__
#define __LEAF_H__
#include "utilites.h"

using namespace std;

class Leaf
{
	public:
			Leaf();
			Leaf(vector<vector<double> >* data_,vector<int>* answ_,vector<int>* items_,int n_1_);
               ~Leaf();

			Leaf* pLeft;    
			Leaf* pRight;   
	
			double PrL,PrR; 
			double impurity;
			double prc;
			double err_prc;
			
			int classify(vector<double>& sample);
			int size() {return size_;}
			int T(int n_leafs=0);

			void split();
			void join();

			void print_leaf(int leaf_level = 0);
			void print_tree(int leaf_level = 0);

	private:	
			pair<double,int> s;
			vector<int>*  items;

			vector<vector<double> >* data;
			vector<int>* answ;

		     int   label; 
			int   size_;
			int   err_n;
			int     n_1; 			
};

#endif

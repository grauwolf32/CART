#include "leaf.h"

using namespace std;

Leaf::Leaf()
{
	pLeft  = NULL;
	pRight = NULL;
	label  = -1; 

	PrL = PrR = 0.0;
	impurity = -1.0;
	n_1 = 0;

	s = pair<double,int>(0.0,-1);
	data = NULL;
	items = NULL;
}

Leaf::Leaf(train_data* data_,vector<int>* items_,int n_1_)
{
	pLeft  = NULL;
	pRight = NULL;
	label  = -1; 

	PrL = PrR = 0.0;
	if( (double)n_1_ / ((double)items_->size()) > 0.5) label = 0;  
	else label = 1;

	impurity = gini_impurity((double)(*data_).size(),(double)n_1_); 
	s = pair<double,int>(0.0,-1);

	n_1 = n_1_;
	items = items_;
	data  = data_; 		
}

Leaf::~Leaf()
{
	pLeft  = NULL;
	pRight = NULL;
	data   = NULL;

	items->clear();

     if(items != NULL)delete items;
}

int Leaf::classify(vector<double>& sample)
{
	if(s.second != -1)
	{
		if(sample[s.second] <= s.first)
		{
			if(pLeft != NULL)
			{ 
				return pLeft->classify(sample);
			}
		}

		else if(pRight != NULL)
		{
			return pRight->classify(sample);
		}
	}
	return label;
}

void Leaf::split()
{
	int m = (int)items->size();
	int k = (*data)[0].second.size();
     int index = 0;

	if( (double)m / data->size() < 0.001 ) {return; }
     if(n_1 <= 0 || n_1 >= (int)items->size()){return;}

	int n_l = 0;
	int n_l_1 = 0;

	int n_l_ = 0;
	int n_l_1_ = 0;

	feature f(m);
	double  imp = 0.0;

	double  best_sep_value = 0.0;

	for(int j = 0; j < k;j++)
	{	
		n_l = 0;
		n_l_1 = 0;

		for(int i = 0; i < m;i++)
		{
			f[i].first  = (*data)[(*items)[i]].second[j];
			f[i].second = (*data)[(*items)[i]].first;
		}

		sort(f.begin(),f.end());

		if(is_equ(f[0].first,f[m-1].first))
		{
			continue;
		}

		for(int i = 0; i < m;i++) 
		{
			n_l += 1;
			if(f[i].second == 0) 
				n_l_1 += 1;
               
			imp = d_imp(m,n_l,n_1,n_l_1);
			
			if(imp > s.first)
			{
				s.first = imp;
				s.second = j;
				n_l_ = n_l;
				n_l_1_ = n_l_1;
                    index = i;
				best_sep_value = f[index].first;
			}	
		}
	}
	s = pair<double,int>(best_sep_value,s.second);
	if(s.second == -1)return;
	
	vector<int>* item_left  = new vector<int>;
	vector<int>* item_right = new vector<int>;

	for(int i = 0;i < m;i++)
	{
		if((*data)[(*items)[i]].second[s.second] <= s.first)
		{
			item_left->push_back((*items)[i]);
		}
		else item_right->push_back((*items)[i]);
	}	

	/*items->clear();*/

	n_l_1 = n_l_1_;
	int n_r_1 = n_1 - n_l_1;

	if(item_left->size() == 0 || item_right->size() == 0)
		{return;}
		
	pLeft = new Leaf(data,item_left,n_l_1);
	pRight = new Leaf(data,item_right,n_r_1);
	
     
	pLeft->split();
	pRight->split();
	
}

void Leaf::print_leaf(int leaf_level)
{
	int n = (int)items->size();
	cout << "\nLeaf in level: "<< leaf_level << "\n";
	cout << "Label : " << label << " %: "<< ((double)n_1)/(double)n<< " n: "<<n <<" n_1: "<<n_1<<"\n";
     cout << "Impurity: "<< impurity << "\n";
     cout << "Separate by value "<<s.first <<" of feature "<<s.second<<"\n";

	if(n > 0)
	{
     	cout << "Elements id in leaf: \n";
     	for(int i = 0;i < n;i++)
     	{
			cout << (*items)[i] << " ";
		}
     	cout << "\n";
	}

	if(s.second != -1 && n > 0)
	{
     	cout << "Values in leaf: \n";
     	for(int i = 0;i < n;i++)
     	{
			cout << (*data)[(*items)[i]].second[s.second] << " ";
		}
    	 	cout << "\n";
	}
}

void Leaf::print_tree(int leaf_level)
{
	print_leaf(leaf_level);
	leaf_level += 1;
	if(pLeft != NULL)
		pLeft->print_tree(leaf_level);
	if(pRight != NULL)
		pRight->print_tree(leaf_level);
}


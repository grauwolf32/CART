#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>

using namespace std;

typedef vector<pair<int,vector<double> > > train_data;
typedef vector<pair<double,int> > 		      feature;

double gini_impurity(double n,double n1);
double d_imp(int n,int n_l, int n_1, int n_1_l);

class Leaf
{
	public:
			Leaf();
			Leaf(train_data* data,vector<int>* items,int n_1);
			~Leaf();

			Leaf* pLeft;    // pointers to the left
			Leaf* pRight;   //  and right child 
	
			int label;      // label of the dominant class in the leaf	
			double PrL,PrR; // procrent fraction 
			double impurity;
			
			int classify(vector<double>& sample);

			void split();
			void destroy();
			void prunge();
			
	private:	
			pair<double,int> s;
			vector<int>*  items;
			train_data*   data;
			int    n_1; // number of items of first class

				
};

Leaf::Leaf()
{
	pLeft  = NULL;
	pRight = NULL;
	label  = -1; // Unlabeled

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
	label  = -1; // Unlabeled

	PrL = PrR = 0.0;
	if( (double)n_1_ / (double)items->size() > 0.5) label = 0; // 1st class
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
	if(items != NULL)delete items;
	items->clear();
}

int Leaf::classify(vector<double>& sample)
{
	if(s.second != -1)
	{
		if(sample[s.second] < s.first)
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

	if( (double)m / data->size() < 0.01 ) return; // Xopow 

	int n_l = 0;
	int n_l_1 = 0;

	int n_l_ = 0;
	int n_l_1_ = 0;

	feature f(m);
	double  imp = 0.0;

	for(int j = 0; j < k;j++)
	{	
		for(int i = 0; i < m;i++)
		{
			f[i].first  = (*data)[(*items)[i]].second[j];
			f[i].second = (*data)[(*items)[i]].first;
		}

		sort(f.begin(),f.end());

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
			}	
		}
	}
	
	vector<int>* item_left = new vector<int>;
	vector<int>* item_right = new vector<int>;

	for(int i = 0;i < m;i++)
	{
		if((*data)[(*items)[i]].second[s.second] < s.first) 
			item_left->push_back((*items)[i]);
		else item_right->push_back((*items)[i]);
	}	

	n_l_1 = n_l_1_;
	int n_r_1 = n_1 - n_l_1;

	pLeft = new Leaf(data,item_left,n_l_1);
	pRight = new Leaf(data,item_right,n_r_1);

	pLeft->split();
	pRight->split();
	
}

void Leaf::destroy()
{
	if(pLeft != NULL)
		pLeft->destroy();
	if(pRight != NULL)
		pRight->destroy();

	delete this; //Possible error
}

void Leaf::prunge()
{
	if(pLeft != NULL)
		pLeft->destroy();
	if(pRight != NULL)
		pRight->destroy();
}

double gini_impurity(double n,double n1)
{	
   return 1.0 - (n1/n)*(n1/n) - ((n-n1)/n)*((n-n1)/n);
}

double d_imp(int n,int n_l, int n_1, int n_1_l) // value to maximize
{
	double n_r = n - n_l;
	double n_2_l = n_l - n_1_l;
	double n_1_r = n_1 - n_1_l;
	double n_2_r = n_r - n_1_r;
	
	return (1.0/(double)n_l)*(double)( (n_1_l)*(n_1_l) + (n_2_l)*(n_2_l) )\
		+ (1.0/(double)n_r)*(double)( (n_1_r)*(n_1_r) + (n_2_r)*(n_2_r) );  
}	

class CART_binar_classifier
{
	public:
			CART_binar_classifier();
			~CART_binar_classifier();

			void train(vector<vector<double> >* data,vector<int>* answ_);
               void print_tree();

			int 	predict(vector<double>& sample);
			Leaf* main_node;
	private:
			int m,n;
			void clear();
			vector<vector<double> >* data_;
			vector<int>* 			answ_;
			train_data 			data;
	
			
};

CART_binar_classifier::CART_binar_classifier()
{
	data_ = NULL;
	answ_ = NULL;
	m = 0;
	n = 0;
}

CART_binar_classifier::~CART_binar_classifier()
{
	clear();
}

void CART_binar_classifier::clear()
{
	//if(answ_ != NULL)delete [] answ_; // Have i done this ?
	//if(data_  != NULL)delete [] data_; //
	if(main_node != NULL)main_node->destroy();

	answ_ = NULL;
	data_ = NULL;
     main_node = NULL;

     m = 0;
     n = 0;

	data.clear();
}

int CART_binar_classifier::predict(vector<double>& sample)
{
	return main_node->classify(sample);
}

void CART_binar_classifier::train(vector<vector<double> >* data_t,vector<int>* answ_t)
{
	clear();

	data_ = data_t;
	answ_ = answ_t;

	m = (int)data_->size();
     n = (int)(*data_)[0].size();
	int n_1 = 0;

	for(int i = 0;i < m;i++)
	{
			data.push_back(pair<int,vector<double> >((*answ_)[i],(*data_)[i]));
	}
     //
     vector<int>* items = new vector<int>;
	for(int i = 0;i < m;i++)
	{
		items->push_back(i);
		if((*answ_t)[i] == 0)n_1 += 1;
	}

	main_node = new Leaf(&data,items,n_1);
	main_node->split();

	return;
}

int main(void)
{
	CART_binar_classifier cart;
	vector<vector<double> >* A = new vector<vector<double> > (100);
	vector<int>* B = new vector<int>;
	vector<double> temp;

	for(int i = 0;i < 100;i++)
	{
		temp[0] = (double)i;
		A[i].push_back(temp);
	 	B->push_back(i%2);
	}
	cart.train(A,B);
	temp[0] = 3;
	cout << cart.predict(temp) << "\n";
	
	return 0;
}

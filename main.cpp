#include <algorithm>
#include <utility>
#include <vector>

using namespace std;

typedef vector<pair<int,vector<double> > > train_data;
typedef vector<pair<double,int> > 		    feature;

double gini_impurity(double n,double n1);
double d_imp(int n,int n_l, int n_1, int n_1_l);

class Leaf
{
	public:
			Leaf();
			Leaf(train_data* data,vector<int>& items,int n_1);
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
			vector<int>  items;
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
}

Leaf::Leaf(train_data* data_,vector<int>& items_,int n_1_)
{
	pLeft  = NULL;
	pRight = NULL;
	label  = -1; // Unlabeled

	PrL = PrR = 0.0;
	if( (double)n_1_ / (double)items.size() > 0.5) label = 0; // 1st class
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
	items.clear();
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
	int m = (int)items.size();
	int k = (*data)[0].second.size();

	if( (double)m / data->size() < 0.1 ) return; // Xopow 

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
			f[i].first  = (*data)[items[i]].second[j];
			f[i].second = (*data)[items[i]].first;
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
	
	vector<int> item_left;
	vector<int> item_right;

	for(int i = 0;i < m;i++)
	{
		if((*data)[items[i]].second[s.second] < s.first) 
			item_left.push_back(items[i]);
		else item_right.push_back(items[i]);
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

	delete this;
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

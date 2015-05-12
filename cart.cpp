#include "cart.h"

using namespace std;

CART_binar_classifier::CART_binar_classifier()
{
	data_ = NULL;
	answ_ = NULL;
     main_node = NULL;

	m = 0;
	n = 0;
}

CART_binar_classifier::~CART_binar_classifier()
{
	clear();
}

int CART_binar_classifier::predict(vector<double>& sample)
{
	return main_node->classify(sample);
}

void CART_binar_classifier::clear()
{
	if(main_node != NULL)
	{
		destroy_tree(main_node);
	}

	answ_ = NULL;
	data_ = NULL;
     main_node = NULL;

     m = 0;
     n = 0;

	data.clear();
}

void CART_binar_classifier::destroy_tree(Leaf* pCurrent)
{
    if(pCurrent == NULL)return;

    Leaf* pLeft = pCurrent->pLeft;
    Leaf* pRight = pCurrent->pRight;

    destroy_tree(pLeft);
    destroy_tree(pRight);

    delete pCurrent;
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

     vector<int>* items = new vector<int>;
	for(int i = 0;i < m;i++)
	{
		items->push_back(i);
		if((*answ_t)[i] == 0)n_1 += 1; 
	}

	main_node = new Leaf(&data,items,n_1);
	main_node->split();

	/*
	data_->clear();
	answ_->clear();
	*/

	return;
}

void CART_binar_classifier::predict(vector<vector<double> >* data_in,vector<int>* predicted)
{
	int n = data_in->size();
	int tmp = 0;

     predicted->clear();

	for(int i = 0;i < n;i++)
	{
		tmp = predict((*data_in)[i]);
		predicted->push_back(tmp);
	}
}

void CART_binar_classifier::print_tree()
{
	main_node->print_tree();
}

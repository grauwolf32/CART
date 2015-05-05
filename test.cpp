#define N 100
#include <iostream>
using namespace std;

class foo{
	public:
		int A[100];
		void destroy();
};

void foo::destroy()
{
	cout << "It works!" << "\n";
	delete this;
}

int main(void)
{
	foo* a = new foo;
	a->destroy();
	return 0;
}

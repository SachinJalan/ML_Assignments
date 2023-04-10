Theoritical time complexity analysis for decision tree making. For Discrete Input and Discrete Output. The algorithm ID3 finds the attribute from attributes which best classifies examples which will take a total time of $N*M$ in the first iteration. Now for binary features the decision tree would be a binary tree and at each level of the tree the total time complexity would be $N*M$ as at each level if we assume that the split of N would be equal then each time at each child would be $\frac{N*M}{2}$. Hence if the total depth of the tree is $d$ then the time complexity would be $O(d*N*M)$ and d would be upper bounded by $log_2 N$

The time complexity for the discrete input real output would be the same as discrete input discrete output because the same procedure is followed in discrete real instead of entropy we use parameter like variance hence the time complexity for this operation will remain $N$ and hence the overall time complexity would also remain $O(d*N*M)$

The time complexity for real input discrete output would be $O(N^2 *M)$ as first if we have an attribute and y in finding the optimal split point first we will sort the array which will take $O(N log_2N)$ time and then fir finding the optimal split point at worst we will have to check $N$ split points and checking each split point would take $O(N)$ time hence the time complexity for this step would be $O(N*log_2 N + N^2)$ dropping the term $N*log_2 N$ the time complexity would be $O(N^2)$ for this step and considering the binary tree the subsequent time complexities would be $ O(N^2 * M) $ , $ O(\frac {N^2 * M}{2}) $,$ O(\frac {N^2 * M}{4}) $ and so on which we can approximate to $O(N^2M)$

The time complexity for the real input and real output case would be same as that in the case of real input discrete output as in the case of real input real output there would be $N^2$ checks and hence the time complexity would be the same as that of real input discrete output case.
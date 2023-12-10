# IE517 - Machine Learning in Finance Laboratory @ UIUC - MS in Financial Engineering
IE517 - Machine Learning in Finance Laboratory Research Project - We replicate the techniques used in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4000756 to estimate american call option price obtained from Heston Stochastic Volatility model using PDE-based techniques with deep neural networks


Workers 

- Kamin Atsavasirilert (MSFE student @ UIUC)
- Ruichen Zhao (MSFE student @ UIUC)

Our contribution

-We show that Gradient Boosting Neural Networks can further improve the performance from the paper (plain neural network), reducing the Mean Absolute Error (from 2.63 cents to 2.14 cents) while sacrificing a little amount of prediction time assuming the prediction is done sequentially. The drawback however can be eliminated using parallel programming.

Reference

Anderson, D. and Ulrych, U., 2022. Accelerated American option pricing with deep neural networks. Swiss Finance Institute Research Paper, (22-03).
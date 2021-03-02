# Robust-Optimization

#参考文献：
1. The Effects of Errors in Means, Variances and Covariances on Optimal Portfolio Choice: 研究误差在均值、方差、协方差上对于优化结果的影响。本研究中进行了复现，结果并不完全符合。
2. Incorporating estimation errors into portfolio selection, Robust portfolio: 稳健优化（Robust Optimization）的理论推导及实践。本研究中使用了改动后的目标函数

#数据文件：Data
1. Data_DIJA: 参考文献1中所用数据，用于论文复现
2. DBC, VOO, etc.: 9个US ETF每日价格数据，用于回测
3. hw6data.mat: 根据已知均值向量与已知协方差矩阵模拟的仿真收益数据。
【关于为什么使用仿真数据？
由于稳健优化主要针对均值向量上误差的敏感性，所使用的协方差矩阵应该是真实的协方差矩阵】

#代码：
1. Effect of Error in Means, Variances and Covariances – Cvxopt: 参考文献1的复现。【此版代码存在瑕疵，当改动协方差矩阵时作为新的输入前没有判断协方差矩阵的半正定性。但由于之后没有再用到这份数据，因此只在Sensitivity Analysis on Markowitz and Robust Optimization中做了更新与调整】  
2. Sensitivity Analysis on Markowitz and Robust Optimization: 在不同风险容忍系数下传统Markowitz均值方差优化与稳健优化结果对于均值误差的敏感度  
3. Robust Optimization Efficient Frontier_simulated data: 演示参考文献2中 真实边界与估计边界在Markowitz均值方差优化与稳健优化下的距离
4. Optimization Backtest: 分别利用Markowitz均值方差优化与稳健优化构建投资组合并进行回测，对投资组合的表现进行可视化展示
5. Optimization_function: 利用Markowitz均值方差优化与稳健优化计算权重的函数

#研究成果：
Asset Allocation Optimization Project Final Report

#ifndef CLUSTER_H
#define CLUSTER_H
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "common.cuh"


struct CLUSTER
{
	//由输入的input_point自动确定大小，或人工指定
	INT_VECTOR grid_dimension;
	INT_VECTOR grid_dimension_minus_one;
	int layer_numbers;
	int grid_numbers;

	std::unordered_map<int, int> remain_point_serial;//记录当前还有多少point未被染色，由grid_serial索引
	std::unordered_map<int, int>point_status;//记录每个point当前的状态（属于哪一个团簇），由grid_serial索引
	int cluster_numbers;
	const int max_cluster_numbers=16;//允许存储的最大团簇数目
	std::vector<std::vector<INT_VECTOR>>cluster;//记录每个团簇及其所述point的位置

	void Initial(const INT_VECTOR grid_dimension, const std::vector<INT_VECTOR> input_point_int_crd);//要求int坐标均为非负数（正常要求）

	//构造标准团簇，两个point连接的定义为考虑26个邻居格点但是排除其中8个顶点格点的情况
	//只要有通路连接两个point，则其属于同一个cluster
	//如果一个团簇的point数目<=2，则不考虑
	void Build_Standard_Cluster();

	//在当前每个cluster的point基础上，每个point都向外扩展26个邻居格点作为扩增后的cluster信息
	void Cluster_Standard_Extend();

	//临时使用的变量
	std::vector<int> current_point;//当前团簇搜索时还需要考虑的point队列
};

#endif //CLUSTER_H

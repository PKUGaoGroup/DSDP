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
	//�������input_point�Զ�ȷ����С�����˹�ָ��
	INT_VECTOR grid_dimension;
	INT_VECTOR grid_dimension_minus_one;
	int layer_numbers;
	int grid_numbers;

	std::unordered_map<int, int> remain_point_serial;//��¼��ǰ���ж���pointδ��Ⱦɫ����grid_serial����
	std::unordered_map<int, int>point_status;//��¼ÿ��point��ǰ��״̬��������һ���Ŵأ�����grid_serial����
	int cluster_numbers;
	const int max_cluster_numbers=16;//����洢������Ŵ���Ŀ
	std::vector<std::vector<INT_VECTOR>>cluster;//��¼ÿ���Ŵؼ�������point��λ��

	void Initial(const INT_VECTOR grid_dimension, const std::vector<INT_VECTOR> input_point_int_crd);//Ҫ��int�����Ϊ�Ǹ���������Ҫ��

	//�����׼�Ŵأ�����point���ӵĶ���Ϊ����26���ھӸ�㵫���ų�����8������������
	//ֻҪ��ͨ·��������point����������ͬһ��cluster
	//���һ���Ŵص�point��Ŀ<=2���򲻿���
	void Build_Standard_Cluster();

	//�ڵ�ǰÿ��cluster��point�����ϣ�ÿ��point��������չ26���ھӸ����Ϊ�������cluster��Ϣ
	void Cluster_Standard_Extend();

	//��ʱʹ�õı���
	std::vector<int> current_point;//��ǰ�Ŵ�����ʱ����Ҫ���ǵ�point����
};

#endif //CLUSTER_H

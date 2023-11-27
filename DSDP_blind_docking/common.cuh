#ifndef COMMON_CUH
#define COMMON_CUH
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//在进行近邻表相关空间网格划分时，假设任意网格内的蛋白原子数小于MAX_ATOM_NUMBERS_IN_NEIGHBOR_GRID_BUCKET
//atom_serial[0]应存储该网格内含有的原子数，使用循环for(int i=1;i<atom_serial[0];i=i+1)来获取网格内原子编号
#define MAX_ATOM_NUMBERS_IN_NEIGHBOR_GRID_BUCKET 64
struct NEIGHBOR_GRID_BUCKET
{
	int atom_serial[MAX_ATOM_NUMBERS_IN_NEIGHBOR_GRID_BUCKET];
};

//对一系列搜索结果，记录其序号id和对应获得的打分energy
//用cmp进行打分从低到高排序（越低越好）。
struct INT_FLOAT
{
	int id;
	float energy;
};
bool cmp(INT_FLOAT& a, INT_FLOAT& b);

struct UINT2
{
	unsigned int a;
	unsigned int b;
};
struct VECTOR {
	float x;
	float y;
	float z;
};
struct INT_VECTOR {
	int int_x;
	int int_y;
	int int_z;
};

//check
struct VECTOR_INT {
	float x;
	float y;
	float z;
	int type;
};

//使用广义坐标描述体系（不存在柔性环）中二面角自由度所用的单个节点node信息
struct NODE
{
	float matrix[9];//绕轴转动矩阵，转动轴方向为n，转动角度为theta
					//注意每时每刻更新n,theta->matrix。

	int root_atom_serial;//该二面角自由度中父节点对应的原子序号
	int branch_atom_serial;//该节点对应的原子序号（即绕着该节点与其父节点定义的直线转动）
	VECTOR a0, n0, a, n;//a代表root_atom_serial原子的坐标
						//n代表从root_atom指向branch_atom的方向矢量
						//a0，n0代表参考构象；a，n代表当前构象（注意每时每刻更新）
	int last_node_serial;//该节点的父节点的编号，如果为-1，则代表该节点不存在父节点
};


//Vina 力场参数
//vina atom type
// X-Score
//const sz XS_TYPE_C_H = 0;
//const sz XS_TYPE_C_P = 1;
//const sz XS_TYPE_N_P = 2;
//const sz XS_TYPE_N_D = 3;
//const sz XS_TYPE_N_A = 4;
//const sz XS_TYPE_N_DA = 5;
//const sz XS_TYPE_O_P = 6;
//const sz XS_TYPE_O_D = 7;
//const sz XS_TYPE_O_A = 8;
//const sz XS_TYPE_O_DA = 9;
//const sz XS_TYPE_S_P = 10;
//const sz XS_TYPE_P_P = 11;
//const sz XS_TYPE_F_H = 12;
//const sz XS_TYPE_Cl_H = 13;
//const sz XS_TYPE_Br_H = 14;
//const sz XS_TYPE_I_H = 15;
//const sz XS_TYPE_Met_D = 16;
//const sz XS_TYPE_SIZE = 17;
#define k_gauss1  -0.035579f
#define k_gauss1_2  4.f

#define k_gauss2  -0.005156f
#define k_gauss2_2  0.25f
#define k_gauss2_c  3.f

#define k_repulsion  +0.840245f

#define k_hydrophobic  -0.035069f
#define k_hydrophobic_a  0.5f
#define k_hydrophobic_b  1.5f
#define k_hydrophobic_ua  1.f
#define k_hydrophobic_ub  0.f

#define k_h_bond  -0.587439f
#define k_h_bond_a  -0.7f
#define k_h_bond_b  0.f
#define k_h_bond_ua  1.f
#define k_h_bond_ub  0.f
struct VINA_ATOM
{
	VECTOR crd;//原子坐标
	int atom_type;//原子类型，默认为18种
	float radius;//原子半径，计算Vina打分时需要
	int is_hydrophobic;//=1 代表疏水原子， =0代表亲水原子
	int is_donor;//=1代表供体
	int is_acceptor;//=1代表受体

	//float charge;//如有需要，可自行添加额外力场函数，如静电相互作用
};







FILE* fopen_safely(const char* file_name, const char* mode);

//check 批量改为distance？
float real_distance(VECTOR& crd, VECTOR& crd2);

//返回三个欧拉角（绕固定的x、y、z轴旋转）
//利用此返回角度进行转动，可保证转动构象与参考构象无关（统计意义上）
VECTOR unifom_rand_Euler_angles();


//计算非氢原子的rmsd，a,b为两个坐标
float calcualte_heavy_atom_rmsd(const int atom_numbers, const VECTOR* a, const VECTOR* b, const int* atomic_number);

int Get_Atomic_Number_From_PDBQT_Atom_Name(const char* atom_name);

//能读取PDBQT文件中的一行（以ATOM开头的行），将原子坐标、电荷、类型补加到crd,charge,atom_type中。
void Read_Atom_Line_In_PDBQT(const char* line, std::vector<VECTOR>& crd, std::vector<float>& charge, std::vector<int>& atom_type);
void Read_Atom_Line_In_MOL2(const char* line, std::vector<VECTOR>& crd, std::vector<float>& charge);

//构建ligand-ligand相互作用的近邻表
//ligand内部近邻表需要剔除这些内容：原子自身、1-2bond相连、1-3angle、1-4dihedral、所有氢元素、属于同一个刚体内部的（同一个节点）
//neighbor_list是记录的剔除后剩余的需要计算的分子内相互作用pair（具体使用可见Build_Inner_Neighbor_List内的注释）
//initial_crd为初始坐标（或某个合理的构象），用于帮助判断哪些原子有1-2bond相连。
//atom_node_serial用于剔除属于同一个刚体内部的相互作用
void Build_Inner_Neighbor_List
(const int atom_numbers, int* neighbor_list, std::vector<VECTOR>& initial_crd, std::vector<int>& atomic_number,
	std::vector<int>& atom_node_serial);

//利用PDBQT中的原子类型atom_type、初始结构initial_crd（合理结构即可）、原子序数atomic_number可计算出对应的Vina打分函数中的原子类型vina_atom
int Build_Vina_Atom(VINA_ATOM* vina_atom, std::vector<int>& atom_type, std::vector<VECTOR>& initial_crd, std::vector<int>& atomic_number);

float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b);
float2 Vina_Pair_Interaction(VINA_ATOM a, VINA_ATOM b, const float dr);//在计算a,b相互作用时用外部给定的距离dr
#endif//COMMON_CUH

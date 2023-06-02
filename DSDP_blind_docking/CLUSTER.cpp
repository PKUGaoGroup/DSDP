#include "CLUSTER.h"
void CLUSTER::Initial(const INT_VECTOR grid_dimension, const std::vector<INT_VECTOR> input_point_int_crd)
{
	//��ʼ��������Ϣ������ӳ��
	this->grid_dimension = grid_dimension;
	grid_dimension_minus_one.int_x = grid_dimension.int_x - 1;
	grid_dimension_minus_one.int_y = grid_dimension.int_y - 1;
	grid_dimension_minus_one.int_z = grid_dimension.int_z - 1;

	layer_numbers = grid_dimension.int_x * grid_dimension.int_y;
	grid_numbers = layer_numbers * grid_dimension.int_z;

	//��ʼ����ǰʣ��δ����point����Ϣ������point��δ����,valueֵ����ν����δ��ʹ�ã�
	//��ÿ��pointһ��Ψһserial���Ҷ�����-1��ŵ��Ŵ�
	remain_point_serial.clear();
	point_status.clear();
	int grid_serial;
	for (int i = 0; i < input_point_int_crd.size(); i += 1)
	{
		grid_serial = input_point_int_crd[i].int_z * layer_numbers + input_point_int_crd[i].int_y * this->grid_dimension.int_x + input_point_int_crd[i].int_x;
		remain_point_serial.insert({ grid_serial,i });
		point_status.insert({ grid_serial,-1 });
	}

	cluster_numbers = 0;
	if (cluster.size() == 0)
	{
		cluster.resize(max_cluster_numbers);
	}
}

void CLUSTER::Build_Standard_Cluster()
{
	current_point.clear();
	int grid_serial;
	INT_VECTOR point_position;

	//�������δȾɫ��ԭ�ӣ��򲻶�ѭ��
	while (remain_point_serial.size() > 0)
	{
		grid_serial = remain_point_serial.begin()->first;
		current_point.push_back(grid_serial);//��ʣ��δȾɫpoint�ĵ�һ����ʼ������current_point��������ÿ�ο�ʼʱ���Ѿ���ɿյ���
		point_status.at(grid_serial) = cluster_numbers;//����ǰpointȾɫ
		remain_point_serial.erase(grid_serial);

		//���ڵ�ǰ�Ŵص�point��δ�������ʱ��
		while (current_point.size() > 0)
		{
			grid_serial = current_point[0];
			point_position.int_x = grid_serial % grid_dimension.int_x;
			point_position.int_y = (grid_serial % layer_numbers)/grid_dimension.int_x;
			point_position.int_z = grid_serial / layer_numbers;

			//ѭ��27���ھӣ�����8�����㣩
			int x, y, z;
			int dx=0, dy=0, dz=0;
			int ddx, ddy, ddz,temp_int;
			for (int neighbor = 0; neighbor < 27; neighbor += 1)
			{
				ddx = dx - 1;
				ddy = dy - 1;
				ddz = dz - 1;

				//���ddx������������һ��Ϊ0����һ������8��������
				if (((ddx & ddy) & ddz) == 0)
				{
					z = (point_position.int_z + ddz);
					y = (point_position.int_y + ddy);
					x = (point_position.int_x + ddx);

					x = x - (x & ((x) >> 31));//���x<0��x=0�����x>=0��x=x
					temp_int = (grid_dimension_minus_one.int_x - x);
					x = x + (temp_int & (temp_int >> 31));//���x<grid_dimension.int_x��x=x�����x>=grid_dimension.int_x��x=grid_dimension_minus_one.int_x

					y = y - (y & ((y) >> 31));
					temp_int = (grid_dimension_minus_one.int_y - y);
					y = y + (temp_int & (temp_int >> 31));

					z = z - (z & ((z) >> 31));
					temp_int = (grid_dimension_minus_one.int_z - z);
					z = z + (temp_int & (temp_int >> 31));

					grid_serial = z * layer_numbers + y * grid_dimension.int_x + x;
					//������ھ�ȷʵ����
					std::unordered_map<int, int>::iterator point_in_point_status_ite = point_status.find(grid_serial);
					if (point_status.end() != point_in_point_status_ite)
					{
						//�����point��δ��Ⱦɫ
						if (point_in_point_status_ite->second == -1)
						{
							point_status.at(grid_serial) = cluster_numbers;
							remain_point_serial.erase(grid_serial);
							current_point.push_back(grid_serial);
						}
					}
				}

				dx = dx + 1;
				dy = dy + (((2 - dx) >> 31) & 0x00000001);
				dx = dx & ((dx - 3) >> 31);
				dz = dz + (((2 - dy) >> 31) & 0x00000001);
				dy = dy & ((dy - 3) >> 31);
			}//for 27

			current_point[0] = current_point[current_point.size()-1];
			current_point.pop_back();
		}//while current_point.size() > 0
		cluster_numbers += 1;
	}
	
	int selected_cluster_numbers = 0;
	for (int i = 0; i < cluster_numbers; i = i + 1)
	{
		//count point numbers;
		int point_numbers = 0;
		for (std::unordered_map<int, int>::iterator ite = point_status.begin(); ite != point_status.end(); ite++)
		{
			if (ite->second == i)
			{
				point_numbers++;
			}
		}
		if (point_numbers > 2&& selected_cluster_numbers<max_cluster_numbers)
		{
			cluster[selected_cluster_numbers].clear();
			for (std::unordered_map<int, int>::iterator ite = point_status.begin(); ite != point_status.end(); ite++)
			{
				if (ite->second == i)
				{
					grid_serial = ite->first;
					point_position.int_x = grid_serial % grid_dimension.int_x;
					point_position.int_y = (grid_serial % layer_numbers) / grid_dimension.int_x;
					point_position.int_z = grid_serial / layer_numbers;
					cluster[selected_cluster_numbers].push_back(point_position);
				}
			}
			selected_cluster_numbers++;
		}
	}
	cluster_numbers = selected_cluster_numbers;
}

void CLUSTER::Cluster_Standard_Extend()
{
	int grid_serial;
	INT_VECTOR point_position;
	for (int cluster_i = 0; cluster_i < cluster_numbers; cluster_i += 1)
	{
		remain_point_serial.clear();//remain_point_serial�ڸô�������Ϊ��¼�����Ŵ�point�ı���
		for (int point_i = 0; point_i < cluster[cluster_i].size(); point_i += 1)
		{
			point_position = cluster[cluster_i][point_i];
			grid_serial = point_position.int_z * layer_numbers + point_position.int_y * grid_dimension.int_x + point_position.int_x;
			remain_point_serial.insert({ grid_serial,1 });//valueû����

			int x, y, z;
			int dx = 0, dy = 0, dz = 0;
			int temp_int;
			for (int neighbor = 0; neighbor < 27; neighbor += 1)
			{
				z = (point_position.int_z + dz-1);
				y = (point_position.int_y + dy-1);
				x = (point_position.int_x + dx-1);

				x = x - (x & ((x) >> 31));//���x<0��x=0�����x>=0��x=x
				temp_int = (grid_dimension_minus_one.int_x - x);
				x = x + (temp_int & (temp_int >> 31));//���x<grid_dimension.int_x��x=x�����x>=grid_dimension.int_x��x=grid_dimension_minus_one.int_x

				y = y - (y & ((y) >> 31));
				temp_int = (grid_dimension_minus_one.int_y - y);
				y = y + (temp_int & (temp_int >> 31));

				z = z - (z & ((z) >> 31));
				temp_int = (grid_dimension_minus_one.int_z - z);
				z = z + (temp_int & (temp_int >> 31));

				grid_serial = z * layer_numbers + y * grid_dimension.int_x + x;
				remain_point_serial.insert({ grid_serial,1 });//valueû����

				dx = dx + 1;
				dy = dy + (((2 - dx) >> 31) & 0x00000001);
				dx = dx & ((dx - 3) >> 31);
				dz = dz + (((2 - dy) >> 31) & 0x00000001);
				dy = dy & ((dy - 3) >> 31);
			}//for 27
		}
		cluster[cluster_i].clear();
		for (std::unordered_map<int, int>::iterator ite = remain_point_serial.begin(); ite != remain_point_serial.end(); ite++)
		{
			grid_serial = ite->first;
			point_position.int_x = grid_serial % grid_dimension.int_x;
			point_position.int_y = (grid_serial % layer_numbers) / grid_dimension.int_x;
			point_position.int_z = grid_serial / layer_numbers;
			cluster[cluster_i].push_back(point_position);
		}
	}
}
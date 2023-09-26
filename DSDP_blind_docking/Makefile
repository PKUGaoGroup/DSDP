

SUBDIRS=$(shell ls -l | grep ^d | awk '{print $$9}')
compiler=nvcc
COMMON_COMMAND=-arch=sm_60 -rdc=true -lcudadevrt -lcufft --use_fast_math -O4 -std=c++17 -lgomp
BIN_NAME=DSDP

install: main_SingleRun_SitePointStart.o common.o CLUSTER.o Copy_pdbqt_Format.o DSDP_Sort.o DSDP_Task.o Kernel.o Neighbor_Grid.o Partial_Rigid_Small_Molecule.o Rigid_Protein.o Vina_Grid_Force_Field.o Site_Information.o 
	$(compiler) -o $(BIN_NAME)  $^ $(COMMON_COMMAND)


all: main_SingleRun_SitePointStart.o common.o CLUSTER.o Copy_pdbqt_Format.o DSDP_Sort.o DSDP_Task.o Kernel.o Neighbor_Grid.o Partial_Rigid_Small_Molecule.o Rigid_Protein.o Vina_Grid_Force_Field.o Site_Information.o
	


main_SingleRun_SitePointStart.o: main_SingleRun_SitePointStart.cu 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


common.o: common.cu common.cuh
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)
	
CLUSTER.o: CLUSTER.cpp CLUSTER.h 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)

Copy_pdbqt_Format.o: Copy_pdbqt_Format.cu common.cuh Copy_pdbqt_Format.cuh
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)

DSDP_Sort.o: DSDP_Sort.cu common.cuh DSDP_Sort.cuh
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


DSDP_Task.o: DSDP_Task.cu DSDP_Task.cuh common.cuh 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


Kernel.o: Kernel.cu Kernel.cuh common.cuh 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


Neighbor_Grid.o: Neighbor_Grid.cu Neighbor_Grid.cuh common.cuh
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


Partial_Rigid_Small_Molecule.o: Partial_Rigid_Small_Molecule.cu Partial_Rigid_Small_Molecule.cuh common.cuh 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


Rigid_Protein.o: Rigid_Protein.cu Rigid_Protein.cuh common.cuh 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)


Vina_Grid_Force_Field.o: Vina_Grid_Force_Field.cu Vina_Grid_Force_Field.cuh common.cuh 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)
	

Site_Information.o: Site_Information.cu Site_Information.cuh common.cuh 
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)

clean:
	rm -f *.o
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.o)

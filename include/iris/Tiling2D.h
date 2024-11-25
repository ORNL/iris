/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 
#pragma once
#ifdef __cplusplus
//#include <cstddef>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include "iris/iris.h"
using namespace std;
#define GET2D_INDEX(I, J, COLS)    ((I)*(COLS)+(J))
#define GET4D_INDEX(I, J, K, L, N, P, Q)    ((I)*(N)*(P)*(Q) + (J)*(P)*(Q) + (K)*(Q)+(L))  
#define GET2D_DATA(DATA, I, J, COLS)    (((DATA)) ? ((DATA)+GET2D_INDEX(I,J,COLS)) : NULL)
#define GET4D_DATA(DATA, I, J, K, L, N, P, Q)    (((DATA)) ? ((DATA)+GET4D_INDEX(I,J,K,L,N,P,Q)) : NULL)
#define DEFAULT_FLATTENED_DATA_FLAG false
namespace iris {
    //#define ENABLE_PIN_MEMORY
    enum Tile2DIteratorType
    {
        TILE2D_COL_FIRST,
        TILE2D_ROW_FIRST,
        TILE2D_RIGHT_DOWN_TREE_WISE,
    };
    template <typename T, typename C>
        class Tiling2DIteratorType
        {
            public:
                Tiling2DIteratorType(C & collection,
                        size_t const row_tile_index, size_t const col_tile_index, 
                        size_t const row_tiles_count, size_t const col_tiles_count,
                        Tile2DIteratorType iterator_type): 
                    row_tile_index(row_tile_index), col_tile_index(col_tile_index),
                    row_tiles_count(row_tiles_count), col_tiles_count(col_tiles_count),
                    collection(collection), iterator_type(iterator_type),
                    row_start_index(row_tile_index), col_start_index(col_tile_index) {
                        state = 0; level = 0;
                    }

                bool operator!= (Tiling2DIteratorType const & other) const
                {
                    return (row_tile_index != other.row_tile_index) && (col_tile_index != other.col_tile_index);
                }

                T & operator* () const
                {
                    return collection.GetAt(row_tile_index, col_tile_index);
                }

                Tiling2DIteratorType const & operator++ ()
                {
                    if (iterator_type == TILE2D_ROW_FIRST) {
                        row_tile_index++;
                        col_tile_index = (row_tile_index==row_tiles_count) ? col_tile_index+1 : col_tile_index;
                        row_tile_index = (row_tile_index==row_tiles_count) ? row_start_index : row_tile_index;
                    } else if (iterator_type == TILE2D_RIGHT_DOWN_TREE_WISE) { 
                        if (state == 0) {
                            // Column index increment
                            col_tile_index++;
                            state = (col_tile_index == col_tiles_count) ? 1 : 0;
                            col_tile_index = (state) ? level : col_tile_index;
                            row_tile_index = (state) ? row_tile_index+1 : row_tile_index;
                        }
                        else if (state == 1) {
                            // Row index increment
                            row_tile_index++;
                            state = (row_tile_index == row_tiles_count) ? 0 : 1;
                            row_tile_index = (state) ? row_tile_index : level+1;
                            col_tile_index = (state) ? col_tile_index : level+1;
                            level = (state) ? level : level+1; 
                        }
                    } else {
                        // TILE2D_COL_FIRST
                        col_tile_index++;
                        row_tile_index = (col_tile_index==col_tiles_count) ? row_tile_index+1 : row_tile_index;
                        col_tile_index = (col_tile_index==col_tiles_count) ? col_start_index : col_tile_index;
                    }
                    return *this;
                }
                Tiling2DIteratorType & operator++ (int index)
                {
                    if (iterator_type == TILE2D_ROW_FIRST) {
                        row_tile_index++;
                        col_tile_index = (row_tile_index==row_tiles_count) ? col_tile_index+1 : col_tile_index;
                        row_tile_index = (row_tile_index==row_tiles_count) ? row_start_index : row_tile_index;
                    } else if (iterator_type == TILE2D_RIGHT_DOWN_TREE_WISE) { 
                        if (state == 0) {
                            // Column index increment
                            col_tile_index++;
                            state = (col_tile_index == col_tiles_count) ? 1 : 0;
                            col_tile_index = (state) ? level : col_tile_index;
                            row_tile_index = (state) ? row_tile_index+1 : row_tile_index;
                        }
                        else if (state == 1) {
                            // Row index increment
                            row_tile_index++;
                            state = (row_tile_index == row_tiles_count) ? 0 : 1;
                            row_tile_index = (state) ? row_tile_index : level+1;
                            col_tile_index = (state) ? col_tile_index : level+1;
                            level = (state) ? level : level+1; 
                        }
                    } else {
                        // TILE2D_COL_FIRST
                        col_tile_index++;
                        row_tile_index = (col_tile_index==col_tiles_count) ? row_tile_index+1 : row_tile_index;
                        col_tile_index = (col_tile_index==col_tiles_count) ? col_start_index : col_tile_index;
                    }
                    return *this;
                }

            private:
                size_t   row_tile_index, col_tile_index;
                size_t   row_tiles_count, col_tiles_count;
                size_t   state, level;
                size_t   row_start_index, col_start_index;
                Tile2DIteratorType iterator_type;
                C & collection;
        };
    template <typename DType>
        class Tiling2D;
    template<typename DType, typename Tiling2D>
        class Tile2DType
        {
            public:
                Tile2DType() { }
                void Init(size_t row, size_t col, Tiling2D *tiling) {
                    row_tile_index_ = row;
                    col_tile_index_ = col;
                    tiling_ = tiling;
                }
                void Print() {
                    printf("Inside tile Row:%d Col:%d\n", row_tile_index_, col_tile_index_);
                }
                DType *Host() {
                    return tiling_->Host(row_tile_index_, col_tile_index_);
                }
                iris_mem IRISMem() {
                    return tiling_->IRISMem(row_tile_index_, col_tile_index_);
                }
                size_t row_tile_index() { return row_tile_index_; }
                size_t col_tile_index() { return col_tile_index_; }
                size_t row_tile_size() { return tiling_->row_tile_size(); }
                size_t col_tile_size() { return tiling_->col_tile_size(); }
            private:
                size_t row_tile_index_;
                size_t col_tile_index_;
                Tiling2D *tiling_;
        };
    template <typename DType>
        using Tile2D = Tile2DType<DType, Tiling2D<DType> >;
    template <typename DType>
        using Tiling2DIterator = Tiling2DIteratorType<Tile2D<DType>, Tiling2D<DType> >;
#define GET_TILE_COUNT(X, SIZE)     (int)(((X)/(float)(SIZE))+0.5f) 
    // Class: Tiling2D 
    //        Data structure for 
    //              1. managing tiling parameters
    //              2. iterators
    //              3. IRIS memory objects for tiles
    //              4. Introduce task depenendencies
    //        Tiling parameters:
    //              size_t row_size_;
    //              size_t col_size_;
    //              size_t row_stride_;
    //              size_t col_stride_;
    //              size_t row_tile_size_;
    //              size_t col_tile_size_;
    //              size_t row_offset_;
    //              size_t col_offset_;
    //              size_t row_tiles_count_;
    //              size_t col_tiles_count_;
    //        Iterators:
    //              TILE2D_ROW_FIRST (Row first/wise tiles iterator)
    //              TILE2D_COL_FIRST (Column first/wise tiles iterator)
    //              TILE2D_RIGHT_DOWN_TREE_WISE (Column followed by Row in a tree wise tiles iterator)
    template <typename DType>
        class Tiling2D
        {
            public:
                using value_type = Tile2D<DType>;
                using iterator = Tiling2DIterator<DType>;
                using type = Tiling2DIterator<DType>;
                Tiling2D() {
                    data_ = NULL;
                    usm_flag_ = false;
                    flattened_data_ = NULL;
                    flattened_data_flag_ = false;
                    iris_mem_tiles_ = NULL;
                    tiles_ = NULL;
                    iterator_type_ = TILE2D_COL_FIRST;
                    row_size_ = 0;
                    col_size_ = 0;
                    reset_flag_ = false;
                    row_stride_ = 0;
                    col_stride_ = 0;
                    row_tile_size_ = 0;
                    col_tile_size_ = 0;
                    row_offset_ = 0;
                    col_offset_ = 0;
                    row_tiles_count_ = 0;
                    col_tiles_count_ = 0;
                }
                Tiling2D(DType *data, size_t row_size, size_t col_size, 
                        size_t row_tile_size, size_t col_tile_size, 
                        bool flattened_data_flag=DEFAULT_FLATTENED_DATA_FLAG, bool enable_bc=false, bool reset_flag=false, bool usm_flag=false)
                {
                    Init(data, row_size, col_size, row_tile_size, col_tile_size,
                            flattened_data_flag, enable_bc, reset_flag, usm_flag);
                    //printf("RowSize:%ld ColSize:%ld RowTile:%ld ColTile:%ld RowTileCount:%ld ColTileCount:%d\n", row_size_, col_size_, row_tile_size_, col_tile_size_, row_tiles_count_, col_tiles_count_); 
                }
                Tiling2D(DType *data, size_t row_size, size_t col_size, 
                        size_t row_tile_size, size_t col_tile_size, 
                        size_t row_stride, size_t col_stride, 
                        bool flattened_data_flag=DEFAULT_FLATTENED_DATA_FLAG, 
                        bool enable_bc = false,
                        bool reset_flag=false, bool usm_flag=false)
                {
                    Init(data, row_size, col_size, row_tile_size, col_tile_size,
                            row_stride, col_stride, flattened_data_flag, enable_bc, reset_flag, usm_flag);
                }
                void Init(DType *data, size_t row_size, size_t col_size, 
                        size_t row_tile_size, size_t col_tile_size, 
                        bool flattened_data_flag=DEFAULT_FLATTENED_DATA_FLAG, 
                        bool enable_bc = false,
                        bool reset_flag=false, bool usm_flag=false)
                {
                    InitCore(data, row_size, col_size, 
                            row_tile_size, col_tile_size, flattened_data_flag, enable_bc, reset_flag, usm_flag);
                    row_stride_ = row_tile_size;
                    col_stride_ = col_tile_size;
                    row_tiles_count_ = GET_TILE_COUNT(row_size, row_tile_size_);
                    col_tiles_count_ = GET_TILE_COUNT(col_size, col_tile_size_);
                    AllocateTiles();
                    AllocateFlattenedData();
                    AllocateIRISMem();
                    //printf("RowSize:%ld ColSize:%ld RowTile:%ld ColTile:%ld RowTileCount:%ld ColTileCount:%d\n", row_size_, col_size_, row_tile_size_, col_tile_size_, row_tiles_count_, col_tiles_count_); 
                }
                void Init(DType *data, size_t row_size, size_t col_size, 
                        size_t row_tile_size, size_t col_tile_size, 
                        size_t row_stride, size_t col_stride, 
                        bool flattened_data_flag=DEFAULT_FLATTENED_DATA_FLAG, bool enable_bc=false, bool reset_flag=false, bool usm_flag=false) 
                {
                    InitCore(data, row_size, col_size, row_tile_size, col_tile_size, 
                            flattened_data_flag, enable_bc, reset_flag, usm_flag);
                    row_stride_ = row_stride;
                    col_stride_ = col_stride;
                    row_tiles_count_ = GET_TILE_COUNT(row_size, row_tile_size_); //TODO: Update
                    col_tiles_count_ = GET_TILE_COUNT(col_size, col_tile_size_); //TODO: Update
                    AllocateTiles();
                    AllocateFlattenedData();
                    AllocateIRISMem();
                }
                void AllocateTiles()
                {
                    tiles_ = (Tile2D<DType> *)malloc(sizeof(Tile2D<DType>)*row_tiles_count_*col_tiles_count_);
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            tiles_[i*col_tiles_count_+j].Init(i, j, this);
                        }
                    }
                }
                void Print_Bc_Dev() {
                    /*
                    printf ("Block cyclic distribution\n");
                    printf (" \t");
                    for(size_t j=0; j<col_tiles_count_; j++) {
                        printf ("%d\t", j);
                    }
                    printf ("\n");
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        printf ("%d\t", i);
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            printf ("%d\t", iris_data_mem_get_rr_bc_dev(IRISMem(i, j)));
                        }
                        printf ("\n");
                    } */
                }
                void Transpose()
                {
                    assert(row_size_ == col_size_);
                    for(size_t i=0; i<row_size_; i++) {
                        for(size_t j=0; j<i; j++) {
                            DType tmp = data_[GET2D_INDEX(j,i,col_size_)];
                            data_[GET2D_INDEX(j,i,row_size_)] =  data_[GET2D_INDEX(i,j,col_size_)];
                            data_[GET2D_INDEX(i,j,col_size_)] = tmp;
                        }
                    }
                }
                void FlushAll(iris_graph graph, int target_dev, char* tag)
                {
                    char tn1[256];
                    sprintf(tn1, "%s-all-flush-out", tag);
                    iris_task foAll_task; iris_task_create_name(tn1, &foAll_task);
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            iris_task_dmem_flush_out(foAll_task, IRISMem(i,j));

                            printf(" i=%d j=%d  memory[%lu]\n", 
                                    i, j, iris_mem_get_uid(IRISMem(i,j))
                                  );

                        }
                    }
                    iris_graph_task( graph, foAll_task, target_dev, NULL );
                }
                void FlushTaskAll(iris_graph graph, int target_dev, char* tag, iris_task last_task)
                {
                    char tn1[256];
                   for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            sprintf(tn1, "%s-flush-out-mem-%d-%d", tag, i, j);
                            iris_task fo_task; iris_task_create_name(tn1, &fo_task);
                            iris_task_dmem_flush_out(fo_task, IRISMem(i,j));
                            iris_task_depend(fo_task, 1, &last_task);
                            iris_graph_task( graph, fo_task, target_dev, NULL );
                            last_task = fo_task;
                        }
                    }
                }



                void Flat2Tiled()
                {
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            for(size_t k=0; k<row_tile_size_; k++) {
                                DType *flattened_data = GET4D_DATA(flattened_data_, i, j, k, 0, col_tiles_count_, row_tile_size_, col_tile_size_);
                                DType *data = GET2D_DATA(data_, i*row_tile_size_+k, j*col_tile_size_+0, col_size_);
                                memcpy(flattened_data, data, col_tile_size_*sizeof(DType));
                            }
                        }
                    }
                    memcpy(data_, flattened_data_, row_size_*col_size_*sizeof(DType));
                }
                void Tiled2Flat()
                {
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            for(size_t k=0; k<row_tile_size_; k++) {
                                DType *flattened_data = GET4D_DATA(flattened_data_, i, j, k, 0, col_tiles_count_, row_tile_size_, col_tile_size_);
                                DType *data = GET2D_DATA(data_, i*row_tile_size_+k, j*col_tile_size_+0, col_size_);
                                memcpy(data, flattened_data, col_tile_size_*sizeof(DType));
                            }
                        }
                    }
                }
                void ResetNullHost()
                {
                    if (iris_mem_tiles_ == NULL) return;
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            iris_mem mem = iris_mem_tiles_[i*col_tiles_count_+j];
                            iris_data_mem_update(mem, NULL);
                        }
                    }
                }
                void ResetHost(DType *gdata=NULL)
                {
                    if (iris_mem_tiles_ == NULL) return;
                    if (gdata != NULL) { 
                        data_ = gdata;
                        if (flattened_data_flag_) 
                            Flat2Tiled();
                    }
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        for(size_t j=0; j<col_tiles_count_; j++) {
                            iris_mem mem = iris_mem_tiles_[i*col_tiles_count_+j];
                            DType *data = data_;
                            if (data_ != NULL && flattened_data_flag_) {
                                data = GET4D_DATA(data_, i, j, 0, 0, col_tiles_count_, row_tile_size_, col_tile_size_);
                            }
                            iris_data_mem_update(mem, data);
                        }
                    }
                }
                void AllocateIRISMem()
                {
                    iris_mem_tiles_ = (iris_mem *)malloc(sizeof(iris_mem)*row_tiles_count_*col_tiles_count_);
                    if (flattened_data_flag_) {
                        if (data_ != NULL) {
                            for(size_t i=0; i<row_tiles_count_; i++) {
                                for(size_t j=0; j<col_tiles_count_; j++) {
                                    DType *flattened_data = GET4D_DATA(data_, i, j, 0, 0, col_tiles_count_, row_tile_size_, col_tile_size_);
#ifdef IRIS_MEM_HANDLER
                                    iris_mem_create(row_tile_size_ * col_tile_size_ * sizeof(DType), &iris_mem_tiles_[i*col_tiles_count_+j]);
#else
#ifdef ENABLE_PIN_MEMORY
                                    if (flattened_data)
                                        iris_register_pin_memory(flattened_data, row_tile_size_*col_tile_size_*sizeof(DType));
#endif
                                    iris_data_mem_create(&iris_mem_tiles_[i*col_tiles_count_+j], flattened_data, row_tile_size_*col_tile_size_*sizeof(DType));
                                    if (reset_flag_) 
                                        iris_data_mem_init_reset(iris_mem_tiles_[i*col_tiles_count_+j], 1);
#endif
                                    // To be enabled only when flattend flag is enabled
                                    if (usm_flag_)
                                        iris_mem_enable_usm(iris_mem_tiles_[i*col_tiles_count_+j], iris_model_all);
                                }
                            }
                        }
                    }
                    else {
                        size_t l_host_size[] = { col_size_, row_size_};
                        size_t dev_size[] = { col_tile_size_, row_tile_size_ };
#ifdef ENABLE_PIN_MEMORY
                        if (data_)
                            iris_register_pin_memory(data_, row_size_*col_size_*sizeof(DType));
#endif
                        for(size_t i=0; i<row_tiles_count_; i++) {
                            for(size_t j=0; j<col_tiles_count_; j++) {
                                size_t l_offset[] = { j*col_tile_size_, i*row_tile_size_};
                                //printf("Creating datamem tile %d %d\n", i, j);
#ifdef IRIS_MEM_HANDLER
                                iris_mem_create(row_tile_size_ * col_tile_size_ * sizeof(DType), &iris_mem_tiles_[i*col_tiles_count_+j]);
#else
                                //printf(" i:%d j:%d data:%p\n", i, j, data_);
                                iris_data_mem_create_tile(&iris_mem_tiles_[i*col_tiles_count_+j], data_, l_offset, l_host_size, dev_size, sizeof(DType), 2);
                                if (enable_bc_) iris_data_mem_update_bc(iris_mem_tiles_[i*col_tiles_count_+j], enable_bc_, i, j); 
                                if (reset_flag_) 
                                    iris_data_mem_init_reset(iris_mem_tiles_[i*col_tiles_count_+j], 1);
#endif
                            }
                        }
                    }
                }
                void AllocateFlattenedData()
                {
                    if (flattened_data_flag_) {
                        flattened_data_ = (DType *)malloc(sizeof(DType)*row_tiles_count_*col_tiles_count_*row_tile_size_*col_tile_size_);
                        Flat2Tiled();
                        //memset(flattened_data_, 0, sizeof(DType)*row_tiles_count_*col_tiles_count_*row_tile_size_*col_tile_size_);
                    }
                }
                void InitCore(DType *data, size_t row_size, size_t col_size, 
                        size_t row_tile_size, size_t col_tile_size, 
                        bool flattened_data_flag=DEFAULT_FLATTENED_DATA_FLAG, 
                        bool enable_bc = false, 
                        bool reset_flag=false, 
                        bool usm_flag=false)
                {
                    if (usm_flag) flattened_data_flag = true;
                    data_ = data;
                    usm_flag_ = usm_flag;
                    reset_flag_ = reset_flag;
                    flattened_data_flag_ = flattened_data_flag;
                    row_size_ = row_size;
                    col_size_ = col_size;
                    row_tile_size_ = row_tile_size;
                    col_tile_size_ = col_tile_size;
                    row_stride_ = row_tile_size;
                    col_stride_ = col_tile_size;
                    row_offset_ = col_size * sizeof(DType *);
                    col_offset_ = sizeof(DType);
                    flattened_data_ = NULL;
                    iterator_type_ = TILE2D_COL_FIRST;
                    tiles_ = NULL;
                    iris_mem_tiles_ = NULL;
                    enable_bc_ = enable_bc;
                }
                ~Tiling2D() {
                    if (tiles_ != NULL) free(tiles_);
                    if (iris_mem_tiles_ != NULL) free(iris_mem_tiles_);
                }
                Tile2D<DType> & GetAt(size_t row_tile_index, size_t col_tile_index) {
                    return tiles_[row_tile_index * col_tiles_count_ + col_tile_index];
                }
                DType *Host(size_t row_tile_index, size_t col_tile_index) {
                    if (data_ == NULL) return NULL;
                    if (flattened_data_flag_) 
                        return GET4D_DATA(data_, row_tile_index, col_tile_index, 0, 0, col_tiles_count_, row_tile_size_, col_tile_size_);
                    else 
                        return GET2D_DATA(data_, row_tile_index*row_tile_size_, col_tile_index*col_tile_size_, col_size_);
                }
                iris_mem IRISMem(size_t row_tile_index, size_t col_tile_index) {
                    return iris_mem_tiles_[GET2D_INDEX(row_tile_index, col_tile_index, col_tiles_count_)];
                }
                size_t GetSize() const {
                    return row_tiles_count_ * col_tiles_count_;
                }
                void Set_enable_bc(){ enable_bc_ = true;}
                void SetIteratorType(Tile2DIteratorType iterator_type)
                {
                    iterator_type_ = iterator_type;
                }
                inline vector<Tile2D<DType> > & zitems(Tile2DIteratorType it_type=TILE2D_COL_FIRST, int repeat=1)
                {
                    Tiling2DIterator<DType> it = begin(it_type);
                    Tiling2DIterator<DType> it_end = end(it_type);
                    tiles_vector_.clear();
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            tiles_vector_.push_back(*it);
                    }
                    return tiles_vector_;
                }
                inline vector<Tile2D<DType> > & zitems(size_t start_row_index, size_t start_col_index, Tile2DIteratorType it_type=TILE2D_COL_FIRST, int repeat=1)
                {
                    Tiling2DIterator<DType> it = begin(start_row_index, start_col_index, it_type);
                    Tiling2DIterator<DType> it_end = end(it_type);
                    tiles_vector_.clear();
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            tiles_vector_.push_back(*it);
                    }
                    return tiles_vector_;
                }
                inline vector<Tile2D<DType> > items(Tile2DIteratorType it_type=TILE2D_COL_FIRST, int repeat=1)
                {
                    Tiling2DIterator<DType> it = begin(it_type);
                    Tiling2DIterator<DType> it_end = end(it_type);
                    vector<Tile2D<DType>> out;
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            out.push_back(*it);
                    }
                    return out;
                }
                inline vector<Tile2D<DType> > items(size_t start_row_index, size_t start_col_index, Tile2DIteratorType it_type=TILE2D_COL_FIRST, int repeat=1)
                {
                    Tiling2DIterator<DType> it = begin(start_row_index, start_col_index, it_type);
                    Tiling2DIterator<DType> it_end = end(it_type);
                    vector<Tile2D<DType>> out;
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            out.push_back(*it);
                    }
                    return out;
                }
                inline Tiling2DIterator<DType> begin(size_t start_row_index, size_t start_col_index, Tile2DIteratorType it_type)
                {
                    return Tiling2DIterator<DType>(*this, start_row_index, start_col_index, 
                            row_tiles_count_, col_tiles_count_, 
                            it_type);
                }

                inline Tiling2DIterator<DType> begin(Tile2DIteratorType it_type)
                {
                    return Tiling2DIterator<DType>(*this, 0, 0, 
                            row_tiles_count_, col_tiles_count_, 
                            it_type);
                }

                inline Tiling2DIterator<DType> begin()
                {
                    return Tiling2DIterator<DType>(*this, 0, 0, 
                            row_tiles_count_, col_tiles_count_, 
                            iterator_type_);
                }

                inline Tiling2DIterator<DType> end(Tile2DIteratorType it_type)
                {
                    return Tiling2DIterator<DType>(*this, row_tiles_count_, col_tiles_count_,
                            row_tiles_count_, col_tiles_count_, 
                            it_type);
                }
                inline Tiling2DIterator<DType> end()
                {
                    return Tiling2DIterator<DType>(*this, row_tiles_count_, col_tiles_count_,
                            row_tiles_count_, col_tiles_count_, 
                            iterator_type_);
                }
                size_t row_tile_size() { return row_tile_size_; }
                size_t col_tile_size() { return col_tile_size_; }
                size_t row_tiles_count() { return row_tiles_count_; }
                size_t col_tiles_count() { return col_tiles_count_; }
                size_t row_size() { return row_size_; }
                size_t col_size() { return col_size_; }
                DType *GetData() { return data_; }

            private:
                DType *data_;
                DType *flattened_data_;
                bool flattened_data_flag_;
                bool reset_flag_;
                bool usm_flag_;
                iris_mem *iris_mem_tiles_;
                Tile2D<DType> *tiles_;
                Tile2DIteratorType iterator_type_;
                vector<Tile2D<DType>> tiles_vector_;
                size_t row_size_;
                size_t col_size_;
                size_t row_stride_;
                size_t col_stride_;
                size_t row_tile_size_;
                size_t col_tile_size_;
                size_t row_offset_;
                size_t col_offset_;
                size_t row_tiles_count_;
                size_t col_tiles_count_;
                bool enable_bc_;
        };
}
#endif

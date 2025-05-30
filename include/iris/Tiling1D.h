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
#define GET1D_DATA(DATA, I)    ((DATA)+I)
namespace iris {
    enum Tile1DIteratorType
    {
        TILE1D_COL_FIRST,
        TILE1D_ROW_FIRST,
        TILE1D_RIGHT_DOWN_TREE_WISE,
    };
    template <typename T, typename C>
        class Tiling1DIteratorType
        {
            public:
                Tiling1DIteratorType(C & collection,
                        size_t const row_tile_index, 
                        size_t const row_tiles_count, 
                        Tile1DIteratorType iterator_type): 
                    row_tile_index(row_tile_index), 
                    row_tiles_count(row_tiles_count), 
                    collection(collection), iterator_type(iterator_type),
                    row_start_index(row_tile_index) {
                        state = 0; level = 0;
                    }

                bool operator!= (Tiling1DIteratorType const & other) const
                {
                    return (row_tile_index != other.row_tile_index);
                }

                T & operator* () const
                {
                    return collection.GetAt(row_tile_index);
                }

                Tiling1DIteratorType const & operator++ ()
                {
                    row_tile_index = row_tile_index+1;
                    return *this;
                }
                Tiling1DIteratorType & operator++ (int index)
                {
                    row_tile_index++;
                    return *this;
                }

            private:
                size_t   row_tile_index;
                size_t   row_tiles_count;
                size_t   state, level;
                size_t   row_start_index;
                Tile1DIteratorType iterator_type;
                C & collection;
        };
    template <typename DType>
        class Tiling1D;
    template<typename DType, typename Tiling1D>
        class Tile1DType
        {
            public:
                Tile1DType() { }
                void Init(size_t row, Tiling1D *tiling) {
                    row_tile_index_ = row;
                    tiling_ = tiling;
                }
                void Print() {
                    printf("Inside tile Row:%d\n", row_tile_index_);
                }
                DType *Host() {
                    return tiling_->Host(row_tile_index_);
                }
                iris_mem IRISMem() {
                    return tiling_->IRISMem(row_tile_index_);
                }
                size_t row_tile_index() { return row_tile_index_; }
                size_t row_tile_size() { return tiling_->row_tile_size(); }
            private:
                size_t row_tile_index_;
                Tiling1D *tiling_;
        };
    template <typename DType>
        using Tile1D = Tile1DType<DType, Tiling1D<DType> >;
    template <typename DType>
        using Tiling1DIterator = Tiling1DIteratorType<Tile1D<DType>, Tiling1D<DType> >;
#define GET_TILE_COUNT(X, SIZE)     (int)(((X)/(float)(SIZE))+0.5f) 
    // Class: Tiling1D 
    //        Data structure for 
    //              1. managing tiling parameters
    //              2. iterators
    //              3. IRIS memory objects for tiles
    //              4. Introduce task depenendencies
    //        Tiling parameters:
    //              size_t row_size_;
    //              size_t row_stride_;
    //              size_t row_tile_size_;
    //              size_t row_offset_;
    //              size_t row_tiles_count_;
    //        Iterators:
    //              TILE1D_ROW_FIRST (Row first/wise tiles iterator)
    //              TILE1D_COL_FIRST (Column first/wise tiles iterator)
    //              TILE1D_RIGHT_DOWN_TREE_WISE (Column followed by Row in a tree wise tiles iterator)
    template <typename DType>
        class Tiling1D
        {
            public:
                using value_type = Tile1D<DType>;
                using iterator = Tiling1DIterator<DType>;
                using type = Tiling1DIterator<DType>;
                Tiling1D()
                {
                    data_ = NULL;
                    iris_mem_tiles_ = NULL;
                    tiles_ = NULL;
                    reset_flag_ = false;
                    iterator_type_ = TILE1D_COL_FIRST;
                    row_size_ = 0;
                    row_stride_ = 0;
                    row_tile_size_ = 0;
                    row_offset_ = 0;
                    row_tiles_count_ = 0;
                }
                Tiling1D(DType *data, size_t row_size, size_t row_tile_size, bool usm_flag=false, bool reset_flag=false) 
                {
                    Init(data, row_size, row_tile_size, usm_flag, reset_flag);
                    //printf("RowSize:%ld RowTile:%ld RowTileCount:%ld\n", row_size_, row_tile_size_, row_tiles_count_); 
                }
                Tiling1D(DType *data, size_t row_size, size_t row_tile_size, 
                        size_t row_stride, bool usm_flag=false, bool reset_flag=false)
                {
                    Init(data, row_size, row_tile_size, row_stride, usm_flag, reset_flag);
                }
                void Init(DType *data, size_t row_size, size_t row_tile_size, bool usm_flag=false, bool reset_flag=false) 
                {
                    InitCore(data, row_size, row_tile_size, usm_flag, reset_flag);
                    row_stride_ = row_tile_size;
                    row_tiles_count_ = GET_TILE_COUNT(row_size, row_tile_size_);
                    //printf("row tiles count %d\n", row_tiles_count_);
                    AllocateTiles();
                    AllocateIRISMem();
                    //printf("RowSize:%ld RowTile:%ld RowTileCount:%ld\n", row_size_, row_tile_size_, row_tiles_count_); 
                }
                void Init(DType *data, size_t row_size, 
                        size_t row_tile_size, 
                        size_t row_stride, bool usm_flag=false, bool reset_flag=false)
                {
                    InitCore(data, row_size, row_tile_size, usm_flag, reset_flag);
                    row_stride_ = row_stride;
                    row_tiles_count_ = GET_TILE_COUNT(row_size, row_tile_size_); //TODO: Update
                    AllocateTiles();
                    AllocateIRISMem();
                }
                void AllocateTiles()
                {
                    tiles_ = (Tile1D<DType> *)malloc(sizeof(Tile1D<DType>)*row_tiles_count_);
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        tiles_[i].Init(i, this);
                    }
                }
                void ResetNullHost()
                {
                    if (iris_mem_tiles_ == NULL) return;
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        iris_mem mem = iris_mem_tiles_[i];
                        iris_data_mem_update(mem, NULL);
                    }
                }
                void ResetHost(DType *data=NULL)
                {
                    if (iris_mem_tiles_ == NULL) return;
                    if (data == NULL) data = data_;
                    data_ = data;
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        iris_mem mem = iris_mem_tiles_[i];
                        iris_data_mem_update(mem, data_);
                    }
                }
                void AllocateIRISMem()
                {
                    iris_mem_tiles_ = (iris_mem *)malloc(sizeof(iris_mem)*row_tiles_count_);
                    size_t l_host_size[] = { row_size_};
                    size_t dev_size[] = { row_tile_size_ };
#ifdef ENABLE_PIN_MEMORY
                    if (data_)
                        iris_register_pin_memory(data_, row_size_*sizeof(DType));
#endif
                    for(size_t i=0; i<row_tiles_count_; i++) {
                        size_t l_offset[] = { i*row_tile_size_};
#if 0
                        printf("Row tile size  %d, row_tiles_count %d \n", row_tile_size_, row_tiles_count_);
                        printf("Creating datamem tile %d, %d %d %d\n", i, l_offset[0], l_host_size[0], dev_size[0]);
#endif
#ifdef IRIS_MEM_HANDLER
                        iris_mem_create(row_tile_size_ * sizeof(DType), &iris_mem_tiles_[i]);
#else
                        iris_data_mem_create_tile(&iris_mem_tiles_[i], data_, l_offset, l_host_size, dev_size, sizeof(DType), 1);
                        if (reset_flag_) 
                            iris_data_mem_init_reset(iris_mem_tiles_[i], 1);
#endif
                        if (usm_flag_)
                            iris_mem_enable_usm(iris_mem_tiles_[i], iris_model_all);
                    }
                }
                void InitCore(DType *data, size_t row_size,  
                        size_t row_tile_size, bool usm_flag=false, bool reset_flag=false)
                {
                    data_ = data;
                    reset_flag_ = reset_flag;
                    usm_flag_ = usm_flag;
                    row_size_ = row_size;
                    row_tile_size_ = row_tile_size;
                    row_stride_ = row_tile_size;
                    row_offset_ = sizeof(DType);
                    iterator_type_ = TILE1D_COL_FIRST;
                    tiles_ = NULL;
                    iris_mem_tiles_ = NULL;
                }
                ~Tiling1D() {
                    if (tiles_ != NULL) free(tiles_);
                    if (iris_mem_tiles_ != NULL) free(iris_mem_tiles_);
                }
                Tile1D<DType> & GetAt(size_t row_tile_index) {
                    return tiles_[row_tile_index];
                }
                DType *Host(size_t row_tile_index) {
                    return GET1D_DATA(data_, row_tile_index*row_tile_size_);
                }
                iris_mem IRISMem(size_t row_tile_index) {
                    return iris_mem_tiles_[row_tile_index];
                }
                size_t GetSize() const {
                    return row_tiles_count_;
                }
                void SetIteratorType(Tile1DIteratorType iterator_type)
                {
                    iterator_type_ = iterator_type;
                }
                inline vector<Tile1D<DType> > & zitems(Tile1DIteratorType it_type=TILE1D_COL_FIRST, int repeat=1)
                {
                    Tiling1DIterator<DType> it = begin(it_type);
                    Tiling1DIterator<DType> it_end = end(it_type);
                    tiles_vector_.clear();
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            tiles_vector_.push_back(*it);
                    }
                    return tiles_vector_;
                }
                inline vector<Tile1D<DType> > & zitems(size_t start_row_index, Tile1DIteratorType it_type=TILE1D_COL_FIRST, int repeat=1)
                {
                    Tiling1DIterator<DType> it = begin(start_row_index, it_type);
                    Tiling1DIterator<DType> it_end = end(it_type);
                    tiles_vector_.clear();
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            tiles_vector_.push_back(*it);
                    }
                    return tiles_vector_;
                }
                inline vector<Tile1D<DType> > items(Tile1DIteratorType it_type=TILE1D_COL_FIRST, int repeat=1)
                {
                    Tiling1DIterator<DType> it = begin(it_type);
                    Tiling1DIterator<DType> it_end = end(it_type);
                    vector<Tile1D<DType> > out;
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            out.push_back(*it);
                    }
                    return out;
                }
                inline vector<Tile1D<DType> > items(size_t start_row_index, Tile1DIteratorType it_type=TILE1D_COL_FIRST, int repeat=1)
                {
                    Tiling1DIterator<DType> it = begin(start_row_index, it_type);
                    Tiling1DIterator<DType> it_end = end(it_type);
                    vector<Tile1D<DType> > out;
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            out.push_back(*it);
                    }
                    return out;
                }
                inline Tiling1DIterator<DType> begin(size_t start_row_index, Tile1DIteratorType it_type)
                {
                    return Tiling1DIterator<DType>(*this, start_row_index, 
                            row_tiles_count_, 
                            it_type);
                }

                inline Tiling1DIterator<DType> begin(Tile1DIteratorType it_type)
                {
                    return Tiling1DIterator<DType>(*this, 0, 
                            row_tiles_count_, 
                            it_type);
                }

                inline Tiling1DIterator<DType> begin()
                {
                    return Tiling1DIterator<DType>(*this, 0, 
                            row_tiles_count_, 
                            iterator_type_);
                }

                inline Tiling1DIterator<DType> end(Tile1DIteratorType it_type)
                {
                    return Tiling1DIterator<DType>(*this, row_tiles_count_, 
                            row_tiles_count_, 
                            it_type);
                }
                inline Tiling1DIterator<DType> end()
                {
                    return Tiling1DIterator<DType>(*this, row_tiles_count_, 
                            row_tiles_count_, 
                            iterator_type_);
                }
                size_t row_tile_size() { return row_tile_size_; }
                size_t row_tiles_count() { return row_tiles_count_; }
                size_t row_size() { return row_size_; }
                DType *GetData() { return data_; }

            private:
                DType *data_;
                iris_mem *iris_mem_tiles_;
                Tile1D<DType> *tiles_;
                Tile1DIteratorType iterator_type_;
                vector<Tile1D<DType>> tiles_vector_;
                bool usm_flag_;
                bool reset_flag_;
                size_t row_size_;
                size_t row_stride_;
                size_t row_tile_size_;
                size_t row_offset_;
                size_t row_tiles_count_;
        };
}
#endif //__cplusplus


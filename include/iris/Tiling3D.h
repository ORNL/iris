#pragma once

#ifdef __cplusplus
#include <stdlib.h>
#include <vector>
#include "Tiling2D.h"
#include "iris/iris.h"
#define ENABLE_REGIONS
using namespace std;
#define GET3D_INDEX(K, I, J, NI, NJ)    (((K)*(NI)*(NJ))+((I)*(NJ))+(J))
#define GET3D_DATA(DATA, K, I, J, NI, NJ)    ((DATA)+GET3D_INDEX(K,I,J,NI,NJ))
namespace iris {
    enum Tile3DIteratorType
    {
        TILE3D_ZYX, // Default
        TILE3D_ZXY,
        TILE3D_XYZ,
        TILE3D_XZY,
        TILE3D_YZX,
        TILE3D_YXZ,
    };
    template <typename T, typename C>
        class Tiling3DIteratorType
        {
            public:
                Tiling3DIteratorType(C & container,
                        size_t const z_tile_index,  size_t const y_tile_index,  size_t const x_tile_index, 
                        size_t const z_tiles_count, size_t const y_tiles_count, size_t const x_tiles_count,
                        Tile3DIteratorType iterator_type): 
                    z_tile_index(z_tile_index),   y_tile_index(y_tile_index),   x_tile_index(x_tile_index),
                    z_tiles_count(z_tiles_count), y_tiles_count(y_tiles_count), x_tiles_count(x_tiles_count),
                    container(container), iterator_type(iterator_type),
                    z_start_index(z_tile_index), y_start_index(y_tile_index), x_start_index(x_tile_index) {
                        state = 0; level = 0;
                    }
                bool operator!= (Tiling3DIteratorType const & other) const
                {
                    return (z_tile_index != other.z_tile_index) && (y_tile_index != other.y_tile_index) && (x_tile_index != other.x_tile_index);
                }
                T & operator* () const
                {
                    return container.GetAt(z_tile_index, y_tile_index, x_tile_index);
                }
                inline void increment(size_t & z, size_t & y, size_t & x, size_t z_count, size_t y_count, size_t x_count)
                {
                    x++;
                    y = (x==x_count) ? y+1 : y;
                    z = (y==y_count) ? z+1 : z;
                    y = (y==y_count) ? 0 : y;
                    x = (x==x_count) ? 0 : x;
                }
                Tiling3DIteratorType const & operator++ ()
                {
                    Tile3DIteratorType type = iterator_type;
                    if (type == TILE3D_ZYX) {
                        increment(z_tile_index, y_tile_index, x_tile_index, z_tiles_count, y_tiles_count, x_tiles_count);
                    } else if (type == TILE3D_ZXY) {
                        increment(z_tile_index, x_tile_index, y_tile_index, z_tiles_count, x_tiles_count, y_tiles_count);
                    } else if (type == TILE3D_YXZ) {
                        increment(y_tile_index, x_tile_index, z_tile_index, y_tiles_count, x_tiles_count, z_tiles_count);
                    } else if (type == TILE3D_YZX) {
                        increment(y_tile_index, z_tile_index, x_tile_index, y_tiles_count, z_tiles_count, x_tiles_count);
                    } else if (type == TILE3D_XZY) {
                        increment(x_tile_index, z_tile_index, y_tile_index, x_tiles_count, z_tiles_count, y_tiles_count);
                    } else { // TILE3D_XYZ
                        increment(x_tile_index, y_tile_index, z_tile_index, x_tiles_count, y_tiles_count, z_tiles_count);
                    }
                    return *this;
                }
            private:
                size_t   z_tile_index,  y_tile_index,  x_tile_index;
                size_t   z_tiles_count, y_tiles_count, x_tiles_count;
                size_t   state, level;
                size_t   z_start_index, y_start_index, x_start_index;
                Tile3DIteratorType iterator_type;
                C & container;
        };
    template <typename DType>
        class Tiling3D;
    template<typename DType, typename Tiling3D>
        class Tile3DType
        {
            public:
                Tile3DType() { }
                void Init(size_t z, size_t y, size_t x, Tiling3D *tiling) {
                    z_tile_index_ = z;
                    y_tile_index_ = y;
                    x_tile_index_ = x;
                    tiling_ = tiling;
                }
                void Print() {
                    printf("Inside tile Z:%d Y:%d X:%d\n", z_tile_index_, y_tile_index_, x_tile_index_);
                }
                DType *Host() {
                    return tiling_->Host(z_tile_index_, y_tile_index_, x_tile_index_);
                }
                iris_mem IRISMem() {
                    return tiling_->IRISMem(z_tile_index_, y_tile_index_, x_tile_index_);
                }
                size_t z_tile_index() { return z_tile_index_; }
                size_t y_tile_index() { return y_tile_index_; }
                size_t x_tile_index() { return x_tile_index_; }
            private:
                size_t z_tile_index_;
                size_t y_tile_index_;
                size_t x_tile_index_;
                Tiling3D *tiling_;
        };
    template <typename DType>
        using Tile3D = Tile3DType<DType, Tiling3D<DType> >;
    template <typename DType>
        using Tile3DMerge = Tile2DType<DType, Tiling3D<DType> >;
    template <typename DType>
        using Tiling3DIterator = Tiling3DIteratorType<Tile3D<DType>, Tiling3D<DType> >;
#define GET_TILE_COUNT(X, SIZE)     (int)(((X)/(float)(SIZE))+0.5f) 
    template <typename DType>
        class Tiling3D
        {
            public:
                using value_type = Tile3D<DType>;
                using iterator = Tiling3DIterator<DType>;
                using type = Tiling3DIterator<DType>;
                Tiling3D(DType *data, size_t z_size, size_t y_size, size_t x_size, size_t z_tile_size, size_t y_tile_size, size_t x_tile_size, bool outer_dim_regions=false, bool reset_flag=false, bool flattened_data_flag=false, bool usm_flag=false)
                {
                    Init(data, z_size, y_size, x_size, z_tile_size, y_tile_size, x_tile_size, reset_flag, flattened_data_flag, usm_flag);
                    z_stride_ = z_tile_size;
                    y_stride_ = y_tile_size;
                    x_stride_ = x_tile_size;
                    outer_dim_regions_ = outer_dim_regions;
                    z_tiles_count_ = GET_TILE_COUNT(z_size, z_tile_size_);
                    y_tiles_count_ = GET_TILE_COUNT(y_size, y_tile_size_);
                    x_tiles_count_ = GET_TILE_COUNT(x_size, x_tile_size_);
                    AllocateTiles();
                    AllocateIRISMem();
                    //printf("RowSize:%ld ColSize:%ld RowTile:%ld ColTile:%ld RowTileCount:%ld ColTileCount:%d\n", y_size_, x_size_, y_tile_size_, x_tile_size_, y_tiles_count_, x_tiles_count_); 
                }
                Tiling3D(DType *data, size_t z_size, size_t y_size, size_t x_size, size_t z_tile_size, size_t y_tile_size, size_t x_tile_size, size_t z_stride, size_t y_stride, size_t x_stride, bool outer_dim_regions=false, bool reset_flag=false, bool flattened_data_flag=false, bool usm_flag=false) 
                {
                    Init(data, z_size, y_size, x_size, z_tile_size, y_tile_size, x_tile_size, reset_flag, flattened_data_flag, usm_flag);
                    z_stride_ = z_stride;
                    y_stride_ = y_stride;
                    x_stride_ = x_stride;
                    outer_dim_regions_ = outer_dim_regions;
                    z_tiles_count_ = GET_TILE_COUNT(z_size, z_tile_size_); //TODO: Update
                    y_tiles_count_ = GET_TILE_COUNT(y_size, y_tile_size_); //TODO: Update
                    x_tiles_count_ = GET_TILE_COUNT(x_size, x_tile_size_); //TODO: Update
                    AllocateTiles();
                    AllocateIRISMem();
                }
                void AllocateTiles()
                {
                    if (outer_dim_regions_) {
                        tiles_ = (Tile3D<DType> *)malloc(sizeof(Tile3D<DType>)*z_tiles_count_*y_tiles_count_);
                        tiles2d_ = (Tile3DMerge<DType> *)malloc(sizeof(Tile3DMerge<DType>)*z_tiles_count_);
                        for(size_t k=0; k<z_tiles_count_; k++) {
                            tiles2d_[k].Init(k, 0, this);
                            for(size_t i=0; i<y_tiles_count_; i++) {
                                tiles_[k*y_tiles_count_+i].Init(k, i, 0, this);
                            }
                        }

                    }
                    else {
                        tiles_ = (Tile3D<DType> *)malloc(sizeof(Tile3D<DType>)*z_tiles_count_*y_tiles_count_*x_tiles_count_);
                        for(size_t k=0; k<z_tiles_count_; k++) {
                            for(size_t i=0; i<y_tiles_count_; i++) {
                                for(size_t j=0; j<x_tiles_count_; j++) {
                                    tiles_[k*y_tiles_count_*x_tiles_count_+i*x_tiles_count_+j].Init(k, i, j, this);
                                }
                            }
                        }
                    }
                }
                void AllocateIRISMem()
                {
                    if (outer_dim_regions_) {
                        iris_mem_tiles_ = (iris_mem *)malloc(sizeof(iris_mem)*z_tiles_count_*y_tiles_count_);
                        iris_mem_tiles2d_ = (iris_mem *)malloc(sizeof(iris_mem)*z_tiles_count_);
                    }
                    else {
                        iris_mem_tiles_ = (iris_mem *)malloc(sizeof(iris_mem)*z_tiles_count_*y_tiles_count_*x_tiles_count_);
                    }
                    //size_t l_host_size[] = { x_size_, y_size_};
                    size_t l_host_size3d[] = { x_size_, y_size_, z_size_};
                    //size_t dev_size[] = { x_tile_size_, y_tile_size_ };
                    size_t dev_size3d[] = { x_tile_size_, y_tile_size_, z_tile_size_ };
                    size_t bl_host_size[] = { x_tile_size_, y_size_};
                    size_t bl_dev_size[]  = { x_tile_size_, y_size_};
                    size_t bl_offset[] = { 0, 0 };
#ifdef ENABLE_PIN_MEMORY
                    if (data_)
                        iris_register_pin_memory(data_, x_size_ * y_size_ * z_size_ * sizeof(DType));
#endif
                    for(size_t k=0; k<z_tiles_count_; k++) {
                        if (outer_dim_regions_) {
                            iris_mem *mem = &iris_mem_tiles2d_[k];
                            DType *data = GET3D_DATA(data_, k*z_tile_size_, 0, 0, y_size_, x_size_);
                            iris_data_mem_create_tile(mem, data, 
                                    bl_offset, bl_host_size, 
                                    bl_dev_size, sizeof(DType), 2);
                            //printf("Tiling3D: iris_mem_tiles2d_[k:%ld] host:%p z:%ld ysize:%ld xsize:%ld\n", k, data, k*z_tile_size_, y_size_, x_size_);
                            iris_data_mem_enable_outer_dim_regions(*mem);
                            if (reset_flag_) 
                                iris_data_mem_init_reset(*mem, 1);
                            if (usm_flag_) 
                                iris_mem_enable_usm(*mem, iris_model_all);
                        }
                        for(size_t i=0; i<y_tiles_count_; i++) {
                            if (outer_dim_regions_) {
                                //iris_data_mem_create(mem, data+j*x_tile_size_, x_tile_size_*sizeof(DType));
                                DType *data = GET3D_DATA(data_, k*z_tile_size_, i*y_tile_size_, 0, y_size_, x_size_);
                                size_t index2d = GET2D_INDEX(k, i, y_tiles_count_);
                                assert(index2d < z_tiles_count_*y_tiles_count_);
                                iris_mem *mem = &iris_mem_tiles_[index2d];
                                //printf("Tiling3D: iris_mem_tiles_[%ld][%ld] host:%p z:%ld ysize:%ld xsize:%ld\n", k, i, data, k*z_tile_size_, y_size_, x_size_);
                                iris_data_mem_create_region(mem, iris_mem_tiles2d_[k], i);
                                if (usm_flag_) 
                                    iris_mem_enable_usm(*mem, iris_model_all);
                            }
                            else for(size_t j=0; j<x_tiles_count_; j++) {
                                // Convert to 2D tile
                                size_t index3d = GET3D_INDEX(k*z_tile_size_, i*y_tile_size_, j*x_tile_size_, y_size_, x_size_);
                                size_t l_offset3d[] = { j*x_tile_size_, i*y_tile_size_, k*z_tile_size_};
                                //printf("Creating datamem tile %d %d\n", i, j);
                                iris_mem *mem = &iris_mem_tiles_[index3d];
                                iris_data_mem_create_tile(mem, data_, l_offset3d, l_host_size3d, dev_size3d, sizeof(DType), 3);
                                if (reset_flag_) 
                                    iris_data_mem_init_reset(*mem, 1);
                                //Not handled USM for this case
                            }
                        }
                    }
                }
                void Init(DType *data, size_t z_size, size_t y_size, size_t x_size, size_t z_tile_size, size_t y_tile_size, size_t x_tile_size, bool reset_flag, bool flattened_data_flag, bool usm_flag=false)
                {
                    usm_flag_ = usm_flag;
                    if (usm_flag) flattened_data_flag = true;
                    flattened_data_flag_ = flattened_data_flag;
                    reset_flag_ = reset_flag;
                    data_ = data;
                    z_size_ = z_size;
                    y_size_ = y_size;
                    x_size_ = x_size;
                    z_tile_size_ = z_tile_size;
                    y_tile_size_ = y_tile_size;
                    x_tile_size_ = x_tile_size;
                    z_stride_ = z_tile_size;
                    y_stride_ = y_tile_size;
                    x_stride_ = x_tile_size;
                    z_offset_ = y_size * x_size * sizeof(DType *);
                    y_offset_ = x_size * sizeof(DType *);
                    x_offset_ = sizeof(DType);
                    flattened_data_ = NULL;
                    iterator_type_ = TILE3D_ZYX;
                    tiles_ = NULL;
                    tiles2d_ = NULL;
                    iris_mem_tiles_ = NULL;
                    iris_mem_tiles2d_ = NULL;
                }
                ~Tiling3D() {
                    if (tiles_ != NULL) free(tiles_);
                    if (tiles2d_ != NULL) free(tiles2d_);
                    if (iris_mem_tiles_ != NULL) free(iris_mem_tiles_);
                    if (iris_mem_tiles2d_ != NULL) free(iris_mem_tiles2d_);
                }
                Tile3DMerge<DType> & GetMerge(size_t z_tile_index) {
                    //printf("tile2d_[%ld]\n", z_tile_index);
                    return tiles2d_[z_tile_index]; // Must be outer_dim_regions_
                }
                Tile3D<DType> & GetRegion(size_t z_tile_index, size_t y_tile_index) {
                    return tiles_[GET2D_INDEX(z_tile_index, y_tile_index, y_tiles_count_)];
                }
                Tile3D<DType> & GetAt(size_t z_tile_index, size_t y_tile_index, size_t x_tile_index) {
                    return tiles_[GET3D_INDEX(z_tile_index, y_tile_index, x_tile_index, y_tiles_count_, x_tiles_count_)];
                }
                DType *Host(size_t z_tile_index, size_t y_tile_index, size_t x_tile_index) {
                    return GET3D_DATA(data_, z_tile_index*z_tile_size_, y_tile_index*y_tile_size_, x_tile_index*x_tile_size_, y_size_, x_size_);
                }
                iris_mem IRISMem(size_t z_tile_index, size_t y_tile_index) {
                    return iris_mem_tiles2d_[z_tile_index]; // Must be outer_dim_regions_
                }
                iris_mem IRISMem(size_t z_tile_index, size_t y_tile_index, size_t x_tile_index) {
                    if (outer_dim_regions_)
                        return iris_mem_tiles_[GET2D_INDEX(z_tile_index, y_tile_index, y_tiles_count_)];
                    else
                        return iris_mem_tiles_[GET3D_INDEX(z_tile_index, y_tile_index, x_tile_index, y_tiles_count_, x_tiles_count_)];
                }
                size_t GetSize() const {
                    return z_tiles_count_ * y_tiles_count_ * x_tiles_count_;
                }
                void SetIteratorType(Tile3DIteratorType iterator_type)
                {
                    iterator_type_ = iterator_type;
                }
                inline vector<Tile3D<DType> > items(Tile3DIteratorType it_type=TILE3D_ZYX, int repeat=1)
                {
                    Tiling3DIterator<DType> it = begin(it_type);
                    Tiling3DIterator<DType> it_end = end(it_type);
                    vector<Tile3D<DType> > out;
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            out.push_back(*it);
                    }
                    return out;
                }
                inline vector<Tile3D<DType> > items(size_t start_z_index, size_t start_y_index, size_t start_x_index, Tile3DIteratorType it_type=TILE3D_ZYX, int repeat=1)
                {
                    Tiling3DIterator<DType> it = begin(start_z_index, start_y_index, start_x_index, it_type);
                    Tiling3DIterator<DType> it_end = end(it_type);
                    vector<Tile3D<DType> > out;
                    for(; it != it_end; ++it) {
                        for(int i=0; i<repeat; i++)
                            out.push_back(*it);
                    }
                    return out;
                }
                inline Tiling3DIterator<DType> begin(size_t start_z_index, size_t start_y_index, size_t start_x_index, Tile3DIteratorType it_type)
                {
                    return Tiling3DIterator<DType>(*this, start_z_index, start_y_index, start_x_index, 
                            z_tiles_count_, y_tiles_count_, x_tiles_count_, 
                            it_type);
                }

                inline Tiling3DIterator<DType> begin(Tile3DIteratorType it_type)
                {
                    return Tiling3DIterator<DType>(*this, 0, 0, 0,
                            z_tiles_count_, y_tiles_count_, x_tiles_count_, 
                            it_type);
                }

                inline Tiling3DIterator<DType> begin()
                {
                    return Tiling3DIterator<DType>(*this, 0, 0, 0,
                            z_tiles_count_, y_tiles_count_, x_tiles_count_, 
                            iterator_type_);
                }

                inline Tiling3DIterator<DType> end(Tile3DIteratorType it_type)
                {
                    return Tiling3DIterator<DType>(*this, z_tiles_count_, y_tiles_count_, x_tiles_count_,
                            z_tiles_count_, y_tiles_count_, x_tiles_count_, 
                            it_type);
                }
                inline Tiling3DIterator<DType> end()
                {
                    return Tiling3DIterator<DType>(*this, z_tiles_count_, y_tiles_count_, x_tiles_count_,
                            z_tiles_count_, y_tiles_count_, x_tiles_count_, 
                            iterator_type_);
                }
                size_t z_tile_size() { return z_tile_size_; }
                size_t y_tile_size() { return y_tile_size_; }
                size_t x_tile_size() { return x_tile_size_; }
                size_t z_tiles_count() { return z_tiles_count_; }
                size_t y_tiles_count() { return y_tiles_count_; }
                size_t x_tiles_count() { return x_tiles_count_; }
                DType *GetData() { return data_; }

            private:
                DType *data_;
                DType *flattened_data_;
                iris_mem *iris_mem_tiles_;
                iris_mem *iris_mem_tiles2d_;
                bool outer_dim_regions_;
                bool reset_flag_;
                bool usm_flag_;
                bool flattened_data_flag_;
                Tile3D<DType> *tiles_;
                Tile3DMerge<DType> *tiles2d_;
                Tile3DIteratorType iterator_type_;
                size_t z_size_;
                size_t y_size_;
                size_t x_size_;
                size_t z_stride_;
                size_t y_stride_;
                size_t x_stride_;
                size_t z_tile_size_;
                size_t y_tile_size_;
                size_t x_tile_size_;
                size_t z_offset_;
                size_t y_offset_;
                size_t x_offset_;
                size_t z_tiles_count_;
                size_t y_tiles_count_;
                size_t x_tiles_count_;
        };
}
#endif

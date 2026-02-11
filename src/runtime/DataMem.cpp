#include "DataMem.h"
#include "DataMemRegion.h"
#include "Debug.h"
#include "Platform.h"
#include "Device.h"
#include "Task.h"
#include "Utils.h"
#include <stdlib.h>
#include <string>
#include <sstream>

#define USE_MEMRANGE

using namespace std;

namespace iris {
namespace rt {

DataMem::DataMem(Platform* platform, void *host_ptr, size_t size, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) {
    Init(platform, host_ptr, size);
    set_element_type(element_type);
}
DataMem::DataMem(Platform* platform, void *host_ptr, size_t size, const char *symbol, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) {
    Init(platform, host_ptr, size);
    set_element_type(element_type);
    for (int i = 0; i < ndevs_; i++) {
        archs_[i] = archs_dev_[i]->GetSymbol(symbol);
        //printf("---- symbol:%s mem:%lu arch:%p\n", symbol, uid(), archs_[i]);
    }
    is_symbol_ = true;
}
DataMem::DataMem(Platform* platform, void *host_ptr, size_t *host_size, int dim, size_t elem_size, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) {
    size_t size = elem_size;
    ASSERT(dim < DMEM_MAX_DIM);
    for(int i=0; i<dim; i++) {
        size = size * host_size[i];
    }
    Init(platform, host_ptr, size, dim);
    int max_dim = IRIS_MAX4(dim_, host_dim_, dev_dim_, dev_tile_dim_);
    set_element_type(element_type);
    memcpy(dev_size_, host_size, sizeof(size_t)*max_dim);
    memcpy(dev_tile_size_, host_size, sizeof(size_t)*max_dim);
    memcpy(host_size_, host_size, sizeof(size_t)*max_dim);
    memcpy(host_tile_size_, host_size, sizeof(size_t)*max_dim);
    elem_size_ = elem_size;
    compute_strides(host_strides_, host_size_, host_dim_);
    compute_strides(dev_strides_,  dev_size_, dev_dim_);
    //_printf("2DataMem uid:%lu ptr:%p host_ptr:%p host_size:%ld elem_size:%ld dim:%d size:%ld off:%p", uid(), this, host_ptr, host_size[0], elem_size, dim, size, off_);
    //Print();
}
DataMem::DataMem(Platform *platform, void *host_ptr, size_t *off, size_t *host_size, size_t *dev_size, size_t elem_size, int dim, int element_type) : BaseMem(IRIS_DMEM, platform->ndevs()) 
{
    size_t size = elem_size;
    size_t hsize = elem_size;
    for(int i=0; i<dim; i++) {
        size = size * dev_size[i];
        hsize = hsize * host_size[i];
    }
    ASSERT(dim < DMEM_MAX_DIM);
    if (dim == 1) {
        _trace("DataMem host_ptr:%p off:%ld host_size:%ld dev_size:%ld elem_size:%ld dim:%d size:%ld", host_ptr, off[0], host_size[0], dev_size[0], elem_size, dim, size);
    }
    else if (dim == 2) {
        _trace("DataMem host_ptr:%p off:[%ld,%ld] host_size:[%ld,%ld] dev_size:[%ld,%ld] elem_size:%ld dim:%d size:%ld", host_ptr, off[0], off[1], host_size[0], host_size[1], dev_size[0], dev_size[1], elem_size, dim, size);
    }
    else if (dim == 3) {
        _trace("DataMem host_ptr:%p off:[%ld,%ld,%ld] host_size:[%ld,%ld,%ld] dev_size:[%ld,%ld,%ld] elem_size:%ld dim:%d size:%ld", host_ptr, off[0], off[1], off[2], host_size[0], host_size[1], host_size[2], dev_size[0], dev_size[1], dev_size[2], elem_size, dim, size);
    }
    Init(platform, host_ptr, size, dim);
    set_element_type(element_type);
    if (hsize != size)
        tile_enabled_ = true;
    int max_dim = IRIS_MAX4(dim_, host_dim_, dev_dim_, dev_tile_dim_);
    memcpy(off_, off, sizeof(size_t)*max_dim);
    memcpy(dev_size_, dev_size, sizeof(size_t)*max_dim);
    memcpy(dev_tile_size_, dev_size, sizeof(size_t)*max_dim);
    memcpy(host_size_, host_size, sizeof(size_t)*max_dim);
    memcpy(host_tile_size_, dev_size, sizeof(size_t)*max_dim); // This should tile
    compute_strides(host_strides_, host_size_, host_dim_);
    compute_strides(dev_strides_,  dev_size_, dev_dim_);
    elem_size_ = elem_size;
    //Print();
}
void DataMem::compute_strides(size_t *strides, size_t *size, int dim)
{
    if (dim == -1) dim = dim_; 
    strides[0] = 1;
    for(int i=1; i<dim; i++) {
        strides[i] = size[i-1] * strides[i-1];
    }
    strides[dim] = size[dim-1] * strides[dim-1];
}
size_t DataMem::get_alloc_size(size_t *strides) 
{
    if (strides == NULL) strides = dev_strides_;
    size_t size=elem_size_;
    for(int i=1; i<=dim_; i++) {
        size *= strides[i] / strides[i-1];
    }
    return size;
}
void DataMem::compute_dev_strides(size_t *size)
{
    compute_strides(dev_strides_, size, dev_dim_);
}
void DataMem::compute_host_strides(size_t *size)
{
    compute_strides(host_strides_, size, host_dim_);
}
string DataMem::ToString(bool check_regions)
{
    stringstream dev_ptr_addr;
    stringstream dev_ptr;
    stringstream dev_dirty;
    dev_ptr_addr << "[";
    dev_ptr << "[";
    dev_dirty << "[";
    for (int i = 0; i < ndevs_; i++) {
        dev_dirty << is_dev_dirty(i, check_regions);
        dev_ptr_addr << hex << get_arch_ptr(i);
        dev_ptr << hex << get_arch(i);
        if (i != ndevs_-1) {
            dev_dirty << ", ";
            dev_ptr << ", ";
            dev_ptr_addr << ", ";
        }
    }
    dev_dirty << "]";
    dev_ptr << "]";
    dev_ptr_addr << "]";
    string host_tile_size_str = Utils::GetString(host_tile_size_, dim());
    string dev_size_str = Utils::GetString(dev_size_, dim());
    string dev_tile_size_str = Utils::GetString(dev_tile_size_, dim());
    string host_size_str = Utils::GetString(host_size_, dim());
    string off_size_str = Utils::GetString(off_, dim());
    string dev_off_size_str = Utils::GetString(dev_off_, dim());
    string host_strides_size_str = Utils::GetString(host_strides_, dim()+1);
    string dev_strides_size_str = Utils::GetString(dev_strides_, dim()+1);

    stringstream output;
    if (GetMemHandlerType() == IRIS_DMEM_REGION) 
        output << " DMEM-REGION";
    else
        output << " DMEM";
    output << " name:" << name();
    output << " uid:" << uid();
    output << " dim:" << dim_;
    output << " host_dim:" << host_dim_;
    output << " dev_dim:" << dev_dim_;
    output << " dev_tile_dim:" << dev_tile_dim_;
    output << " preserve_size_flag:" << preserve_size_flag_;
    output << " element_size:" << elem_size_;
    if (get_n_regions() > 0) 
        output << " regions:" << get_n_regions();
    output << " host_off:" << off_size_str;
    output << " host_strides:" << host_strides_size_str;
    output << " host_tile_size:" << host_tile_size_str;
    output << " host_size:" << host_size_str;
    output << " dev_tile_size:" << dev_tile_size_str;
    output << " dev_size:" << dev_size_str;
    output << " dev_off:" << dev_off_size_str;
    output << " dev_strides:" << dev_strides_size_str;
    output << " host_ptr:" << host_ptr_;
    output << " dev_ptr:" << dev_ptr.str();
    output << " dev_ptr_addr:" << dev_ptr_addr.str();
    output << " host_dirty:" << is_host_dirty(check_regions);
    output << " dev_dirty:" << dev_dirty.str();
    output << " reset:" << reset_;
    output << " enable_reset:" << enable_reset_;
    output << " reset_type:" << reset_data_.reset_type_;
    if (GetMemHandlerType() == IRIS_DMEM_REGION) {
        DataMemRegion *rmem = (DataMemRegion *)this;
        output << " own_dev_memory:" << rmem->own_dev_memory();
        output << " region:" << rmem->region_index();
        output << " dev_offset_from_root:" << rmem->dev_offset_from_root();
        output << " parent: [";
        output << rmem->get_dmem()->ToString(check_regions); 
        output << " ]";
        if (rmem != rmem->get_clone_region()) {
            output << " clone: [";
            output << " clone-mem:" << rmem->get_clone_region()->ToString(check_regions);
            output << " ]";
        }
    }
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        output << " source: [";
        output << source_mem->ToString(check_regions);
        output << " ]";
    }
    return output.str();
}
void DataMem::Print(bool check_regions)
{
    string output = ToString(check_regions);
    _printf("%s", output.c_str());
}
void DataMem::Print(const char *tag, bool check_regions)
{
    char const_tag[] = "";
    if (tag == NULL) tag = const_tag;
    string output = ToString(check_regions);
    _printf("%s%s", tag, output.c_str());
}
void DataMem::Print(bool check_regions, const char *tag)
{
    char const_tag[] = "";
    if (tag == NULL) tag = const_tag;
    string output = ToString(check_regions);
    _printf("%s%s", tag, output.c_str());
}
DataMem *DataMem::Copy()
{
    DataMem *dmem = new DataMem(platform_, NULL, off_, host_size_, dev_size_, elem_size_, dim_, element_type_);
    dmem->tile_enabled_ = tile_enabled_;
    dmem->dev_dim_ = dev_dim_;
    dmem->dev_tile_dim_ = dev_tile_dim_;
    dmem->host_dim_ = host_dim_;
    return dmem;
}
bool DataMem::is_host_dirty(bool check_regions) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        return source_mem->is_host_dirty(check_regions);
    if (!check_regions || valid_n_regions_ == 0) {
        return host_dirty_flag_;
    } 
    for(int i=0; i<valid_n_regions_; i++) {
        if (regions_[i].first->is_host_dirty())
            return true;
    }
    return false;
}
void DataMem::clear_host_dirty() { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        source_mem->clear_host_dirty();
    else if (valid_n_regions_ == 0) 
        host_dirty_flag_ = false; 
    else {
        host_dirty_flag_ = false; 
        for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->clear_host_dirty();
            // Secondary is not required as the secodary always depends on 
            // first object for host dirty flag management.
        }
    }
}
void DataMem::set_host_dirty(bool flag) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        source_mem->set_host_dirty(flag);
    else if (valid_n_regions_ == 0) 
        host_dirty_flag_ = flag; 
    else {
        host_dirty_flag_ = flag; 
        for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->set_host_dirty(flag);
        }
    }
}
bool DataMem::is_dev_dirty(int devno, bool check_regions) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        return source_mem->is_dev_dirty(devno, check_regions);
    if (!check_regions || valid_n_regions_ == 0) 
        return dirty_flag_[devno]; 
    if (check_regions) for(int i=0; i<valid_n_regions_; i++) {
        if (regions_[i].first->is_dev_dirty(devno))
            return true;
    }
    else
        return dirty_flag_[devno];
    return false;
}
void DataMem::set_dev_dirty(int devno, bool flag, bool check_regions) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        source_mem->set_dev_dirty(devno, flag);
    else if (!check_regions || valid_n_regions_ == 0) 
        dirty_flag_[devno] = flag; 
    else {
        dirty_flag_[devno] = flag; 
        if (check_regions) for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->set_dev_dirty(devno, flag);
            if (regions_[i].first != regions_[i].second) 
                regions_[i].second->set_dev_dirty(devno, flag);
        }
    }
}
int DataMem::get_dev_affinity() { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) return source_mem->get_dev_affinity();
    if (valid_n_regions_ == 0) {
        for(int i=0; i<ndevs_; i++) 
            if (!is_dev_dirty(i)) 
                return i; 
        return -1;
    }
	//TODO : Get the maximum non-dirty flag based device selection
    //_error("This case is not handled");
	//Print(false);
	//Utils::PrintStackTrace();
    return -1;
}
void DataMem::clear_dev_dirty(int devno) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        source_mem->clear_dev_dirty(devno);
    else if (valid_n_regions_ == 0) 
        dirty_flag_[devno] = false; 
    else {
        dirty_flag_[devno] = false; 
        for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->clear_dev_dirty(devno);
            if (regions_[i].first != regions_[i].second) 
                regions_[i].second->clear_dev_dirty(devno);
        }
    }
}
void DataMem::set_dirty_except(int devno, bool check_regions) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        source_mem->set_dirty_except(devno);
    else if (!check_regions || valid_n_regions_ <= 0) {
        for(int i=0; i<ndevs_; i++) {
            if (i != devno) dirty_flag_[i] = true;
        }
        dirty_flag_[devno] = false;
        if (check_regions && GetMemHandlerType() == IRIS_DMEM_REGION) {
            DataMemRegion *rmem = (DataMemRegion *)this;
            if (rmem != rmem->get_clone_region()) {
                rmem->get_clone_region()->set_dirty_all(true, false);
            }
        }
    } 
    else {
        for(int i=0; i<ndevs_; i++) {
            if (i != devno) dirty_flag_[i] = true;
        }
        dirty_flag_[devno] = false;
        if (check_regions) for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->set_dirty_except(devno);
            if (regions_[i].first != regions_[i].second)  {
                // We shouldn't update the device copy as we requested to update the whole host
                //regions_[i].second->set_dirty_except(devno);
                // Set all device copy is invalid
                regions_[i].second->set_dirty_all(true, false);
            }
        }
    }
}
void DataMem::set_dirty_all(bool flag, bool check_regions) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        source_mem->set_dirty_all(flag);
    else if (valid_n_regions_ <= 0) {
        for(int i=0; i<ndevs_; i++) {
            dirty_flag_[i] = flag;
        }
        if (check_regions && GetMemHandlerType() == IRIS_DMEM_REGION) {
            DataMemRegion *rmem = (DataMemRegion *)this;
            if (rmem != rmem->get_clone_region()) {
                rmem->get_clone_region()->set_dirty_all(flag, false);
            }
        }
    } 
    else {
        for(int i=0; i<ndevs_; i++) {
            dirty_flag_[i] = flag;
        }
        for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->set_dirty_all(flag);
            if (regions_[i].first != regions_[i].second) 
                regions_[i].second->set_dirty_all(flag, false);
        }
    }
}
int *DataMem::get_non_dirty_devices(int *dev) { 
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) 
        return source_mem->get_non_dirty_devices(dev);
    else if (valid_n_regions_ == 0) {
        int i=0,j=0;
        for(i=0, j=0; i<ndevs_; i++) {
            if (!dirty_flag_[i]) dev[j++] = i;
        }
        dev[j++] = -1;
        return dev;
    } 
    else {
        ASSERT(0 && "This case is not handled");
        for(int i=0; i<valid_n_regions_; i++) {
            regions_[i].first->get_non_dirty_devices(dev);
            if (regions_[i].first != regions_[i].second) 
                regions_[i].second->get_non_dirty_devices(dev);
        }
    }
    return dev;
}
bool DataMem::is_dirty_all() {
    DataMem *source_mem = (DataMem*)get_source_mem();
    if (source_mem != NULL) return source_mem->is_dirty_all();
    else if (valid_n_regions_ == 0) {
        bool all=true;
        for(int i=0; i<ndevs_; i++) {
            all = all & dirty_flag_[i];
        }
        return all;
    }
    bool all=true;
    for(int i=0; i<valid_n_regions_; i++) {
        all = all & regions_[i].first->is_dirty_all();
        if (regions_[i].first != regions_[i].second) 
            all = all & regions_[i].second->is_dirty_all();
    }
    return all;
}
int DataMem::update_host_offs(size_t *offs, int dim) 
{
    if (dim == -1)
        dim = host_dim_;
    memcpy(off_, offs, sizeof(size_t)*dim);
    for(int i=dim; i<IRIS_MAX4(host_dim_, dev_dim_, dev_tile_dim_, dim_); i++)
        off_[i] = 0;
    return IRIS_SUCCESS;
}
int DataMem::update_host_size(size_t *host_size, bool update_strides, int dim) 
{
    if (dim == -1)
        dim = host_dim_;
    memcpy(host_size_, host_size, sizeof(size_t)*dim);
    for(int i=dim; i<IRIS_MAX4(host_dim_, dev_dim_, dev_tile_dim_, dim_); i++)
        host_size_[i] = 1;
    if (update_strides)
        compute_strides(host_strides_, host_size_, IRIS_MAX5(dim, dev_dim_, dev_tile_dim_, host_dim_, dim_));
    return IRIS_SUCCESS;
}
int DataMem::update_host_tile_size(size_t *host_size, int dim) 
{
    //Print("Before host update:");
    if (dim == -1)
        dim = host_dim_;
    if (dim > 1)
        multi_dim_dev_offset_ = true;
    memcpy(host_tile_size_, host_size, sizeof(size_t)*dim);
    for(int i=dim; i<IRIS_MAX4(host_dim_, dev_dim_, dev_tile_dim_, dim_); i++)
        host_tile_size_[i] = 1;
    //Print("After host update:");
    int n_regions = get_n_regions();
    if (n_regions > 0) {
        vector<pair<DataMemRegion*, DataMemRegion*> > & regions = 
            get_regions();
        for (int i=0; i<n_regions; i++) {
            size_t lhost_size[DMEM_MAX_DIM];
            memcpy(lhost_size, host_size, sizeof(size_t)*dim);
            lhost_size[dim-1] = 1;
            ASSERT(regions[i].first == regions[i].second);
            regions[i].first->update_host_tile_size(lhost_size, dim);
            regions[i].first->enable_multi_dim_dev_offset();
        }
    }
    return IRIS_SUCCESS;
}
int DataMem::update_dev_tile_size(size_t *dev_size, int dim) 
{
    //Print("Before host update:");
    if (dim == -1)
        dim = host_dim_;
    if (dim > 1)
        multi_dim_dev_offset_ = true;
    preserve_size_flag_ = true;
    memcpy(dev_tile_size_, dev_size, sizeof(size_t)*dim);
    for(int i=dim; i<IRIS_MAX4(host_dim_, dev_dim_, dev_tile_dim_, dim_); i++)
        dev_tile_size_[i] = 1;
    //Print("After host update:");
    size_t size = elem_size_;
    for(int i=0; i<dim; i++) {
        size = size * dev_size[i];
    }
    size_ = size;
    //Print("After dev update:");
    set_dirty_all();
    int n_regions = get_n_regions();
    if (n_regions > 0) {
        vector<pair<DataMemRegion*, DataMemRegion*> > & regions = 
            get_regions();
        for (int i=0; i<n_regions; i++) {
            size_t ldev_size[DMEM_MAX_DIM];
            memcpy(ldev_size, dev_size, sizeof(size_t)*dim);
            ldev_size[dim-1] = 1;
            ASSERT(regions[i].first == regions[i].second);
            regions[i].first->update_dev_tile_size(ldev_size, dim);
            regions[i].first->enable_multi_dim_dev_offset();
            regions[i].first->reset_preserve_size_flag();
        }
    }
    return IRIS_SUCCESS;
}
int DataMem::update_dev_size(size_t *dev_size, bool update_strides, int dim) {
    if (dim == -1)
        dim = dim_; 
    if (dim > 1)
        multi_dim_dev_offset_ = true;
    //Print("Before dev update:");
    memcpy(dev_size_, dev_size, sizeof(size_t)*dim); 
    for(int i=dim; i<IRIS_MAX4(host_dim_, dev_dim_, dev_tile_dim_, dim_); i++)
        dev_size_[i] = 1;
    if (update_strides)
        compute_strides(dev_strides_,  dev_size_, IRIS_MAX5(dim, host_dim_, dev_tile_dim_, dev_dim_, dim_));
    size_t size = elem_size_;
    for(int i=0; i<dim; i++) {
        size = size * dev_size[i];
    }
    size_ = size;
    //Print("After dev update:");
    set_dirty_all();
    int n_regions = get_n_regions();
    if (n_regions > 0) {
        vector<pair<DataMemRegion*, DataMemRegion*> > & regions = 
            get_regions();
        for (int i=0; i<n_regions; i++) {
            size_t ldev_size[DMEM_MAX_DIM];
            memcpy(ldev_size, dev_size, sizeof(size_t)*dim);
            ldev_size[dim-1] = 1;
            ASSERT(regions[i].first == regions[i].second);
            regions[i].first->update_dev_size(ldev_size, update_strides, dim);
            regions[i].first->enable_multi_dim_dev_offset();
        }
    }
    return IRIS_SUCCESS;
}
void DataMem::FetchDataFromDevice(void *dst_host_ptr)
{   
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        FetchDataFromDevice(dst_host_ptr);
        return;
    }
    //printf("Fetch mem:%lu dst_host_ptr:%p host_dirty_flag:%d host_ptr:%p\n", uid(), dst_host_ptr, host_dirty_flag_, host_ptr_);
    if (!host_dirty_flag_ && host_ptr_ != NULL) {
        memcpy(dst_host_ptr, host_ptr_, size_);
    }
    else {
        Task* task = Task::Create(platform_, IRIS_TASK, "probe");
        task->set_async(false);
        //printf("Fetch: Mem:%lu Host do not have valid data host:%p size:%lu\n", uid(), host_ptr(), size());
        // No need to submit the task; Just extract data from device
        for (int i = 0; i < ndevs_; i++) {
            if (is_usm(i)) {
                ASSERT(host_ptr_ != NULL);
                memcpy(dst_host_ptr, host_ptr_, size_);
                break;
            }
            if (archs_[i] != NULL && !dirty_flag_[i]) {
                archs_dev_[i]->MemD2H(task, this, off(), 
                        host_size(), dev_size(), elem_size(), 
                        dim(), size(), dst_host_ptr, "Probe");
                break;
            }
        }
        //printf("Completed Fetch: Mem:%lu Host do not have valid data host:%p size:%lu\n", uid(), host_ptr(), size());
        delete task;
    }
}
void DataMem::FetchDataFromDevice(void *dst_host_ptr, size_t size)
{   
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->FetchDataFromDevice(dst_host_ptr, size);
        return;
    }
    //printf("Fetch1 mem:%lu dst_host_ptr:%p host_dirty_flag:%d host_ptr:%p size:%lu\n", uid(), dst_host_ptr, host_dirty_flag_, host_ptr_, size);
    if (!host_dirty_flag_ && host_ptr_ != NULL) {
        memcpy(dst_host_ptr, host_ptr_, size_);
    }
    else {
        size_t host_size[] = { size };
        size_t dev_size[] = { size };
        size_t off[] = { 0 };

        Task* task = Task::Create(platform_, IRIS_TASK, "probe");
        task->set_async(false);
        //printf("Fetch1: Mem:%lu Host do not have valid data host:%p size:%lu\n", uid(), host_ptr(), size);
        // No need to submit the task; Just extract data from device
        for (int i = 0; i < ndevs_; i++) {
            if (is_usm(i)) {
                ASSERT(host_ptr_ != NULL);
                memcpy(dst_host_ptr, host_ptr_, size);
                break;
            }
            if (archs_[i] != NULL && !dirty_flag_[i]) {
                archs_dev_[i]->MemD2H(task, this, off, 
                        host_size, dev_size, elem_size(), 
                        1, size, dst_host_ptr, "Probe");
                break;
            }
        }
        //printf("Completed Fetch: Mem:%lu Host do not have valid data host:%p size:%lu\n", uid(), host_ptr(), size);
        delete task;
    }
}
void DataMem::Init(Platform *platform, void *host_ptr, size_t size, int dim)
{
    preserve_size_flag_ = false;
    dim_ = dim;
    dev_tile_dim_ = dim;
    dev_dim_ = dim;
    host_dim_ = dim;
    multi_dim_dev_offset_ = false;
    source_mem_ = NULL;
    tile_enabled_ = false;
    is_symbol_ = false;
    platform_ = platform;
    host_ptr_owner_ = false;
    size_ = size;
    n_regions_ = 0;
	valid_n_regions_ = 0;
    dynamic_regions_ = false;
    tmp_host_ptr_ = NULL;
    for(size_t i=0; i<regions_.size(); i++) {
        pair<DataMemRegion *, DataMemRegion *> & region = regions_[i]; 
        bool different = region.first != region.second;
        delete region.first;
        if (different) delete region.second;
    }
    regions_.clear();
    host_dirty_flag_ = false;
    is_pin_memory_ = false;
    if (host_ptr == NULL)
        host_dirty_flag_ = true;
    dirty_flag_ = new bool[ndevs_];
    for (int i = 0; i < ndevs_; i++) {
        archs_[i] = NULL;
        archs_dev_[i] = platform->device(i);
        dirty_flag_[i] = true;
        //dev_ranges_[i] = NULL;
    }
    elem_size_ = size_;
    for(int i=0; i<DMEM_MAX_DIM; i++) {
        host_size_[i] = 1;
        dev_size_[i] = 1;
        dev_tile_size_[i] = 1;
        host_tile_size_[i] = 1;
        off_[i] = 0;
        dev_off_[i] = 0;
        host_strides_[i] = dev_strides_[i] = 1;
    }
    host_strides_[dim_] = dev_strides_[dim_] = 1;
    host_ptr_ = host_ptr;
    //printf("host pointer %p\n", host_ptr_);
#ifdef AUTO_PAR
  current_writing_task_ = NULL;
#ifdef AUTO_FLUSH
  flush_task_ = NULL;
#endif
#ifdef AUTO_SHADOW
  host_shadow_dirty_flag_ = true; //initially the shadow is dirty
  current_dmem_shadow_ = NULL; //initially the shadow is dirty
  is_shadow_ = false; // 0: not a shadow, 1: a shadow
  main_dmem_ = NULL; // 0: not a shadow, 1: a shadow
  has_shadow_ = false; // 0: does not have a shadow, 1: has a shadow
#endif
#endif
  bc_ = false;
  row_ = -1;
  col_ = -1;
  rr_bc_dev_ = -1;
  // for keeping track whether any device initiated h2d at the beginning
  for (int i = 0; i < platform_->ndevs(); i++){
    h2d_df_flag_[i] = false;
  }
}
DataMem::~DataMem() {
    //_printf("Releasing memory: %lu host:%p name:%s source_mem:%p\n", uid(), host_ptr(), name(), source_mem_);
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (!platform_->IsFinalized() && is_pin_memory_ && source_mem == NULL) {
        //printf("Unregistering mem:%p\n", host_ptr());
        platform_->DataMemUnRegisterPin(this);
    }
    if (host_ptr_owner_ && source_mem == NULL) free(host_ptr_);
    if (tmp_host_ptr_) free(tmp_host_ptr_);
    if (!platform_->IsFinalized() && source_mem == NULL && 
        GetMemHandlerType() == IRIS_DMEM) {
      for (int i = 0; i < ndevs_; i++) {
        if (archs_[i] && !is_usm(i)) archs_dev_[i]->MemFree(this, archs_[i]);
      }
    }
    delete [] dirty_flag_;
    for(size_t i=0; i<regions_.size(); i++) {
        pair<DataMemRegion *, DataMemRegion *> & region = regions_[i]; 
        bool different = region.first != region.second;
        delete region.first;
        if (different) delete region.second;
    }
    regions_.clear();
}
void DataMem::UpdateHost(void *host_ptr)
{
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->UpdateHost(host_ptr);
        return;
    }
    if (is_pin_memory_ && host_ptr_ != NULL && host_ptr_ != host_ptr) {
        platform_->DataMemUnRegisterPin(this);
    }
    if (host_ptr_owner_ && host_ptr_ != NULL && host_ptr_ != host_ptr) {
        free(host_ptr_);
    }
    void *old_host_ptr = host_ptr_;
    host_ptr_ = host_ptr;
    host_ptr_owner_ = false;
    host_dirty_flag_ = false;
    host_reset_ = false;
    enable_reset_ = false;
    for(int i=0; i<ndevs_; i++) {
        dirty_flag_[i] = true;
    }
    if (is_pin_memory_ && old_host_ptr != host_ptr) platform_->DataMemRegisterPin(this);
}
void DataMem::realllocate_space(size_t elem_size, size_t total)
{
    size_t size = elem_size * total;
    // Supported only for one dim
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->realllocate_space(elem_size, total);
        return;
    }
    elem_size_ = elem_size;
    ASSERT(dim_ == 1);
    ASSERT(valid_n_regions_ <= 0);
    if (size > size_) { 
        if (host_ptr_owner_) free(host_ptr_);
        host_ptr_ = malloc(size);
        host_ptr_owner_ = true;
        host_dirty_flag_ = true;
        for(int i=0; i<ndevs_; i++) {
            if (!dirty_flag_[i]) {
                if (archs_[i] && !is_usm(i)) {
                    archs_dev_[i]->MemFree(this, archs_[i]);
                    archs_[i] = NULL;
                }
                dirty_flag_[i] = true;
            }
        }
    }
    update_host_size(&total);
    update_host_tile_size(&total);
    update_dev_size(&total);
    update_dev_tile_size(&total);
}
void DataMem::CopyToHost(void *host)
{
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->CopyToHost(host);
        return;
    }
    void *host_ptr = host_memory();
    //printf(" Copy to host: %p, host_ptr:%p\n", host, host_ptr);
    memcpy(host_ptr, host, size_);
    host_reset_ = false;
    enable_reset_ = false;
    host_dirty_flag_ = false;
    for(int i=0; i<ndevs_; i++) {
        dirty_flag_[i] = true;
    }
}
void DataMem::RefreshHost()
{
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->RefreshHost();
        return; 
    }
    if (!reset_)
        host_dirty_flag_ = false;
    else
        host_dirty_flag_ = true;
    enable_reset_ = reset_;
    for(int i=0; i<ndevs_; i++) {
        dirty_flag_[i] = true;
    }
}
void DataMem::init_reset(bool reset)
{
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->init_reset(reset);
        return; 
    }
    reset_ = reset;
    enable_reset_ = reset;
    host_dirty_flag_ = reset;
    for(int i=0;  i<ndevs_; i++) {
        dirty_flag_[i] = !reset;
    }
}
void DataMem::AddChild(DataMem *child, size_t offset)
{
  child_.push_back(make_pair(child, offset));
}
void DataMem::clear() {
  DataMem *source_mem = (DataMem *)get_source_mem();
  if (source_mem != NULL) {
      source_mem->clear();
      return; 
  }
  host_dirty_flag_ = false;
  for(int i=0;  i<ndevs_; i++) {
      dirty_flag_[i] = true;
  }
  for (int i = 0; i < ndevs_; i++) {
      if (archs_[i]) {
          if (! is_usm(i)) archs_dev_[i]->MemFree(this, archs_[i]);
          archs_[i] = NULL;
      }
  }
  enable_reset_ = reset_;
}
void *DataMem::tmp_host_memory() {
    if (!tmp_host_ptr_)  {
        tmp_host_ptr_ = malloc(size_);
    }
    return tmp_host_ptr_;
}
void *DataMem::host_memory() {
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) return source_mem->host_memory();
    if (!host_ptr_)  {
        host_ptr_ = malloc(size_); 
        platform_->DataMemRegisterPin(this);
        host_ptr_owner_ = true;
        host_dirty_flag_ = true;
        //printf("host_ptr_owner is enabled:%lu size:%lu\n", uid(), size_);
        if (is_reset()) {
            ResetData & r_data = reset_data();
            if (r_data.reset_type_ == iris_reset_memset) {
                memset(host_ptr_, 0, size_);
                host_dirty_flag_ = false;
                host_reset_ = true;
            }
            else if (element_type_ != iris_unknown) {
                int reset_type = r_data.reset_type_;
                if (reset_type == iris_reset_assign) 
                    Utils::Fill(host_ptr_, size_/elem_size(), element_type_, reset_data_);
                else if (reset_type == iris_reset_arith_seq) 
                    Utils::ArithSequence(host_ptr_, size_/elem_size(), element_type_, reset_data_);
                else if (reset_type == iris_reset_geom_seq) 
                    Utils::GeometricSequence(host_ptr_, size_/elem_size(), element_type_, reset_data_);
                else if (reset_type == iris_reset_random_uniform_seq) 
                    Utils::RandomUniformSeq(host_ptr_, size_/elem_size(), element_type_, reset_data_);
                else {
                    _error("Unknown reset_type: %d\n", reset_type);
                }
                host_dirty_flag_ = false;
                host_reset_ = true;
            }
            else {
                _error("Dont know how to handle element_type:%d\n", element_type_);
            }
        }
    }
    return host_ptr_;
}
void DataMem::EnableOuterDimensionRegions()
{
    vector<pair<DataMemRegion*, DataMemRegion *> > & regions = 
        get_regions();
    if (regions.size() != 0) return;
    int outer_dim = dev_size_[dim_-1];
    set_n_regions(outer_dim);
	set_valid_regions(outer_dim);
    size_t dev_size[DMEM_MAX_DIM], off[DMEM_MAX_DIM], loff[DMEM_MAX_DIM];
    memcpy(dev_size, dev_size_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(off, off_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(loff, off_, sizeof(size_t)*DMEM_MAX_DIM);
    dev_size[dim_-1] = 1;
    size_t dev_offset = 1;
    for(int i=0; i<dim_-1; i++) 
        dev_offset *= dev_size_[i];
    for(int i=0; i<outer_dim; i++) {
        off[dim_-1] = off_[dim_-1]+i; 
        loff[dim_-1] = 0;
        size_t dev_offset_from_root = i * dev_offset * elem_size_;
        DataMemRegion *region = new DataMemRegion(this, i, off, loff, host_size_, dev_size, elem_size_, dim_, dev_offset_from_root);
        stringstream ss;
        ss << name() << "[" << i << "]";
        region->set_name(ss.str());
        region->compute_dev_strides(host_size_);
        region->update_dev_tile_dim(dim_-1);
        regions.push_back(make_pair(region, region));
    }
}

void DataMem::EnableRegions(int axis)
{
    vector<pair<DataMemRegion*, DataMemRegion *> > & regions = 
        get_regions();
    if (regions.size() != 0) return;
    ASSERT(axis < dim_);
    //if (axis == dim_-1) {
    //    EnableOuterDimensionRegions();
    //    return;
    //}
    int outer_dim = dev_size_[axis];
    set_n_regions(outer_dim);
    set_valid_regions(outer_dim);
    size_t dev_size[DMEM_MAX_DIM], off[DMEM_MAX_DIM], loff[DMEM_MAX_DIM];
    size_t dev_off[DMEM_MAX_DIM];
    memcpy(dev_size, dev_size_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(dev_off, off_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(off, off_, sizeof(size_t)*DMEM_MAX_DIM);
    memcpy(loff, off_, sizeof(size_t)*DMEM_MAX_DIM);
    dev_size[axis] = 1;
    int valid_n_regions = get_n_regions();
    for(int i=0; i<valid_n_regions; i++) {
        off[axis] = off_[axis]+i; 
        loff[axis] = 0;
        dev_off[axis] = off_[axis] + i;
        DataMemRegion *full_region = new DataMemRegion(this, i, off, loff, host_size_, dev_size, elem_size_, dim_, 0);
        full_region->set_dev_off(dev_off);
        full_region->enable_multi_dim_dev_offset();
        full_region->compute_dev_strides(host_size_);
        DataMemRegion *region = new DataMemRegion(this, i, off, loff, host_size_, dev_size, elem_size_, dim_, 0);
        stringstream ss;
        ss << name() << "[" << i << "]";
        region->set_name(ss.str());
        region->enable_multi_dim_dev_offset();
        region->set_dirty_all(true, false);
        region->set_own_dev_memory();
        regions.push_back(make_pair(full_region, region));
        //full_region->Print(false);
        //region->Print(false);
    }
}

iris_mem DataMem::CreateRegion(size_t *off, size_t *size)
{
    if (is_within_bounds(off, size)) {
        string off_str = Utils::GetString(off, dim_);
        string size_str = Utils::GetString(size, dim_);
        string goff_str = Utils::GetString(off_, dim_);
        string gsize_str = Utils::GetString(dev_size_, dim_);
        
        _error("Requested region (offset:%s, size: %s) create is out of bounds of global DMEM (offset:%s, size:%s)", off_str.c_str(), size_str.c_str(), goff_str.c_str(), gsize_str.c_str());
        return iris_mem{NULL, 0};
    }
    int index;
    vector<pair<DataMemRegion*, DataMemRegion *> > & regions = 
        get_regions();
    if ((index = check_overlap(off, size)) != -1) {
        DataMemRegion *region = regions[index].second;
        string off_str = Utils::GetString(off, dim_);
        string size_str = Utils::GetString(size, dim_);
        string roff_str = Utils::GetString(region->off(), dim_);
        string rsize_str = Utils::GetString(region->dev_size(), dim_);
        _error("Requested region (offset:%s, size: %s) create overlaps with another region of global DMEM (offset:%s, size:%s)", off_str.c_str(), size_str.c_str(), roff_str.c_str(), rsize_str.c_str());
        return iris_mem{NULL, 0};
    }
    index = static_cast<int>(regions.size());
    DataMemRegion *full_region = new DataMemRegion(this, index, off, off, host_size_, size, elem_size_, dim_, 0);
    full_region->set_dev_off(off);
    full_region->enable_multi_dim_dev_offset();
    full_region->compute_dev_strides(host_size_);
    DataMemRegion *region = new DataMemRegion(this, index, off, off, host_size_, size, elem_size_, dim_, 0);
    region->enable_multi_dim_dev_offset();
    region->set_own_dev_memory();
    dynamic_regions_ = true;
    regions.push_back(make_pair(full_region, region));
    increment_n_regions();
    increment_valid_regions();
    return *(region->struct_obj());
}

int DataMem::check_overlap(size_t *off, size_t *size) {
    vector<pair<DataMemRegion*, DataMemRegion *> > & regions = 
        get_regions();
    int index=0;
    for (const auto& region : regions) {
        if (region.second->overlaps_with(off, size)) {
            return index;
        }
        index++;
    }
    return -1;
}
bool DataMem::is_within_bounds(size_t *off, size_t *size) 
{
    for (int i = 0; i < dim_; ++i) {
        size_t e_end = off[i] + size[i];
        size_t g_end = off_[i] + dev_size_[i];
        if (off[i] >= dev_size_[i] || e_end > g_end) {
            return false;
        }
    }
    return true;
}

void DataMem::create_dev_mem(Device *dev, int devno, void *host)
{
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) {
        source_mem->create_dev_mem(dev, devno, host); 
        return;
    }
    if (is_usm(devno) && dev->is_shared_memory_buffers() && 
            (host != NULL || host_ptr_ != NULL)) {
        archs_[devno] = dev->GetSharedMemPtr((host == NULL) ? host_ptr_ : host, size());
    }
    else {
        set_usm_flag(devno, false);
        size_t size = get_alloc_size(dev_strides_);
        //_printf(" dev:%d mem:%lu size:%lu, %lu allocation", devno, uid(), size, size_);
        dev->MemAlloc(this, archs_ + devno, size, is_reset());
        if (is_reset()) {
            dirty_flag_[devno] = false;
            // It is not required to make host dirty for all reset types. 
            // It is required only for certain reset types such as random.
            // TODO: Think one more time about this.
            if (!host_reset_) host_dirty_flag_ = true;
        }
    }
}

void** DataMem::arch_ptr(Device *dev, void *host) {
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) return source_mem->arch_ptr(dev, host);
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        create_dev_mem(dev, devno, host);
    }
    return &archs_[dev->devno()];
}

void** DataMem::arch_ptr(int devno, void *host) {
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) return source_mem->arch_ptr(devno, host);
    if (archs_[devno] == NULL) {
        Device *dev = archs_dev_[devno];
        create_dev_mem(dev, devno, host);
    }
    return &archs_[devno];
}

void* DataMem::arch(int devno, void *host) {
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) return source_mem->arch(devno, host);
    if (archs_[devno] == NULL) {
        Device *dev = archs_dev_[devno];
        create_dev_mem(dev, devno, host);
    }
    return archs_[devno];
}

void* DataMem::arch(Device* dev, void *host) {
    DataMem *source_mem = (DataMem *)get_source_mem();
    if (source_mem != NULL) return source_mem->arch(dev, host);
    int devno = dev->devno();
    if (archs_[devno] == NULL) {
        create_dev_mem(dev, devno, host);
    }
    return archs_[devno];
}

vector<UncoveredRegion> DataMem::find_uncovered_tiles()
{
    int ndim = dim_;
    size_t *global_shape = host_size();
    size_t totalSize = 1;
    for (int i = 0; i < ndim; ++i) totalSize *= global_shape[i];

    vector<bool> covered(totalSize, false);

    auto get_linear_index = [&](const size_t point[DMEM_MAX_DIM]) -> size_t {
        size_t idx = 0;
        size_t multiplier = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            idx += point[i] * multiplier;
            multiplier *= global_shape[i];
        }
        return idx;
    };

    auto add_vectors = [](const size_t a[DMEM_MAX_DIM], const size_t b[DMEM_MAX_DIM], size_t out[DMEM_MAX_DIM], int ndim) {
        for (int i = 0; i < ndim; ++i)
            out[i] = a[i] + b[i];
    };

    vector<pair<DataMemRegion*, DataMemRegion *> > & regions = 
        get_regions();
    // Mark existing tiles as covered
    for (const auto& tile_pair : regions) {
        auto & tile = tile_pair.second;
        size_t start[DMEM_MAX_DIM], end[DMEM_MAX_DIM];
        tile->start(start);
        tile->end(end);
        size_t point[DMEM_MAX_DIM];
        memcpy(point, start, sizeof(size_t) * ndim);

        auto increment_point = [&](size_t p[DMEM_MAX_DIM], const size_t start[DMEM_MAX_DIM], const size_t end[DMEM_MAX_DIM]) -> bool {
            for (int i = ndim - 1; i >= 0; --i) {
                p[i]++;
                if (p[i] < end[i]) {
                    return true;
                }
                p[i] = start[i];
            }
            return false;
        };

        do {
            covered[get_linear_index(point)] = true;
        } while (increment_point(point, start, end));
    }

    vector<UncoveredRegion> uncovered_tiles;
    size_t point[DMEM_MAX_DIM] = {0};

    auto increment_global_point = [&](size_t p[DMEM_MAX_DIM]) -> bool {
        for (int i = ndim - 1; i >= 0; --i) {
            p[i]++;
            if (p[i] < global_shape[i]) {
                return true;
            }
            p[i] = 0;
        }
        return false;
    };

    do {
        if (!covered[get_linear_index(point)]) {
            size_t offset[DMEM_MAX_DIM];
            size_t size[DMEM_MAX_DIM];
            memcpy(offset, point, sizeof(size_t) * ndim);
            for (int i = 0; i < ndim; ++i) size[i] = 1;

            // Expand along each dimension
            for (int dim = 0; dim < ndim; ++dim) {
                bool can_expand = true;
                while (can_expand) {
                    size_t idx[DMEM_MAX_DIM] = {0};
                    bool all_uncovered = true;
                    do {
                        size_t test_point[DMEM_MAX_DIM];
                        add_vectors(offset, idx, test_point, ndim);
                        test_point[dim] += size[dim];

                        if (test_point[dim] >= global_shape[dim] || covered[get_linear_index(test_point)]) {
                            all_uncovered = false;
                            break;
                        }

                        // Increment idx
                        int carry = 1;
                        for (int d = ndim - 1; d >= 0; --d) {
                            if (d == dim) continue;
                            idx[d] += carry;
                            if (idx[d] < size[d]) {
                                carry = 0;
                                break;
                            } else {
                                idx[d] = 0;
                                carry = 1;
                            }
                        }
                        if (carry == 1) break;
                    } while (true);

                    if (all_uncovered) {
                        size[dim]++;
                    } else {
                        can_expand = false;
                    }
                }
            }

            // Mark this tile covered
            size_t tileEnd[DMEM_MAX_DIM];
            for (int i = 0; i < ndim; ++i)
                tileEnd[i] = offset[i] + size[i];

            size_t fillPoint[DMEM_MAX_DIM];
            memcpy(fillPoint, offset, sizeof(size_t) * ndim);

            auto increment_tile_point = [&](size_t p[DMEM_MAX_DIM]) -> bool {
                for (int i = ndim - 1; i >= 0; --i) {
                    p[i]++;
                    if (p[i] < tileEnd[i]) {
                        return true;
                    }
                    p[i] = offset[i];
                }
                return false;
            };

            do {
                covered[get_linear_index(fillPoint)] = true;
            } while (increment_tile_point(fillPoint));

            uncovered_tiles.emplace_back(offset, size, ndim);
        }
    } while (increment_global_point(point));

    return uncovered_tiles;
}


} /* namespace rt */
} /* namespace iris */


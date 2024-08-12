#include "AutoDAG.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "Debug.h"
#include "Platform.h"
#include "Kernel.h"
#include <stdlib.h>

#include <iostream>

//#define AUTO_FLUSH ON
//#define INCORRECT_AUTO_FLUSH ON

namespace iris {
namespace rt {

AutoDAG::AutoDAG(Platform* platform, bool enable_df_sched){ 
    platform_ = platform;
    current_graph_ = NULL;
#ifdef AUTO_FLUSH
    enable_auto_flush_ = true;
#endif

#ifdef AUTO_SHADOW
  number_of_shadow_ = 0;
#endif
    auto_dep_ = false;
    num_dev_ = platform_->ndevs();
    cur_dev_ = 0;
    enable_df_sched_ = false;
    if (enable_df_sched) {
        enable_df_sched_ = true;
        initalize_h2d_task();
    }
}

void AutoDAG:: initalize_h2d_task(){
    for (int i = 0; i < num_dev_; i++){
        sprintf(tn, "Task-H2D-Dev-%d", num_dev_);
	    Task* task = Task::Create(platform_, IRIS_TASK, tn);
	    task->set_brs_policy(num_dev_);
	    //task_flush = Task::Create(cmd->platform_, IRIS_TASK, tn);
        h2d_task_list_.push_back(task);
    }
}
//AutoDAG::platform_ = NULL;

void AutoDAG::create_dependency(Command* cmd, Task* task, 
		int param_info, BaseMem* mem, 
        Kernel* kernel, int idx) {
    current_kernel_ = kernel;
    current_idx_ = idx;
    current_param_info_ = param_info;
    if(enable_auto_flush_ == true && current_graph_ != NULL && enable_df_sched_ == true){
        if(h2d_task_list_.size()>0){
            for (int i = 0; i < h2d_task_list_.size(); i++){
                current_graph_->AddTask(h2d_task_list_.at(i), h2d_task_list_.at(i)->uid());
            }
        }
    }

#ifdef IGNORE_MANUAL
    set_auto_dep();
#endif

#ifdef SANITY_CHECK
    set_auto_dep();
#endif


#ifdef AUTO_SHADOW

    /*if( mem->GetMemHandlerType() == IRIS_DMEM) {
        if(((DataMem*)mem)->get_is_shadow() == true) printf("----------------no ghapla---------------\n");
        if(((DataMem*)mem)->get_has_shadow() == true) printf("----------------ghapla---------------\n");
    }*/
#endif
    //printf(" Task %s param_info %d\n", task->name(), param_info);
    //printf(" Task %s , mem read count %d\n", task->name(), mem->get_read_task_list()->size());
    //Task* task_prev = NULL;
    if (param_info == iris_w || param_info == iris_rw) {
    	if(mem->get_current_writing_task() != NULL 
		    && mem->get_read_task_list()->size() == 0 ){
    	    if (mem->get_current_writing_task() != task) {
           	    //task->AddDepend(mem->get_current_writing_task());	
                task->AddDepend(mem->get_current_writing_task(), mem->get_current_writing_task()->uid());
                //printf("For %s -> parent %s main w or wr\n", task->name(), mem->get_current_writing_task()->name());
	        }
    	} else if (  mem->get_read_task_list()->size() != 0 ){
#ifndef AUTO_SHADOW
	        create_multi_read_dependency(task, mem);
 	        mem->erase_all_read_task_list();
#else
            if( mem->GetMemHandlerType() == IRIS_DMEM)
	            create_auto_shadow(cmd, task, mem);
            else{
	            create_multi_read_dependency(task, mem);
 	            mem->erase_all_read_task_list();
            }
#endif 
        }
    if (enable_df_sched_){
        df_bc_scheduling(task, ((DataMem*)mem));
        //add_h2d_df_task(((DataMem*)mem);
    }
 
    //printf("Seting current task info %d\n", param_info);
#ifndef AUTO_SHADOW
 	mem->set_current_writing_task(task);
  	task->add_to_write_list(mem);
#else
 	if(mem->GetMemHandlerType() == IRIS_DMEM && ((DataMem*)mem)->get_has_shadow() == true)  {
        ((DataMem*)mem)->get_current_dmem_shadow()->set_current_writing_task(task);
  	    task->add_to_write_list(((DataMem*)mem)->get_current_dmem_shadow());
    }else {
 	    mem->set_current_writing_task(task);
  	    task->add_to_write_list(mem);
    }
#endif

	//task_prev = mem->get_current_writing_task();

#ifdef AUTO_FLUSH
    if( mem->GetMemHandlerType() == IRIS_DMEM)
 	    create_auto_flush(cmd, task, mem);
#endif

    }
    else if (param_info == iris_r) {
    	if(mem->get_current_writing_task() != NULL) {
    	    if (mem->get_current_writing_task() != task) {
                //task->AddDepend(mem->get_current_writing_task());	
                task->AddDepend(mem->get_current_writing_task(), mem->get_current_writing_task()->uid());
                //printf("For %s -> parent %s main r shadow: %d\n", task->name(), mem->get_current_writing_task()->name(), ((DataMem*)mem)->get_is_shadow() );
	        }
	    }
  	    task->add_to_read_list(mem);
	    if (mem->get_current_writing_task() != task) {
  	        mem->add_to_read_task_list(task);
  	        //set_current_writing_task(NULL);
	    }
    }

#ifdef IGNORE_MANUAL
    unset_auto_dep();
#endif

#ifdef SANITY_CHECK
    unset_auto_dep();
#endif
}

void AutoDAG::create_multi_read_dependency(Task* task, 
		BaseMem* mem) {
    std::vector<Task*>* read_list =  
	    mem->get_read_task_list();
    Task* read_task;
    for( int i = 0 ; i < read_list->size(); i++){
        //for( auto & read_task : read_list){
        //for( auto & read_task : mem->get_read_task_list()){
        read_task = read_list->at(i);
        if (read_task != task) {
            //task->AddDepend(read_task);	
            task->AddDepend(read_task, read_task->uid());
            //printf("For %s -> parent %s multiread\n", task->name(), read_task->name());
	    }
    }
}

#ifdef AUTO_SHADOW
void AutoDAG::create_auto_shadow(Command* cmd, Task* task, 
		BaseMem* mem) {
		//BaseMem& mem) {
    //std::cout << "For Shadow creation " << std::endl;
    std::vector<Task*>* read_list =  
	    mem->get_read_task_list();
    //if( read_list->size() < 1) {
    if( read_list->size() < 1 || ((DataMem*)mem)->get_is_shadow() == true) {
	    create_multi_read_dependency(task, mem);
    } else {
        //std::cout << "For " << task->name() << " shadow creation " << std::endl;
        //std::cout << "Host size  " << *(((DataMem*)mem)->host_size()) << std::endl;
        //std::cout << "device size  " << ((DataMem*)mem)->size() << std::endl;
        void* host_ptr_shadow;
        //host_ptr_shadow = malloc(((DataMem*)mem)->size()); // commented out because we dont need
        //((DataMem*)mem)->set_host_ptr_shadow(host_ptr_shadow);
        
        DataMem* mem_shadow = new DataMem(platform_, 
                NULL, mem->size()); // TODO try with NULL
                //host_ptr_shadow, mem->size()); // TODO try with NULL
        //platform_->insert_into_mems(mem_shadow);
        number_of_shadow_++;

        //mem_shadow->set_host_ptr_shadow(((DataMem*)mem)->host_ptr());

        ((DataMem*)mem)->set_current_dmem_shadow(mem_shadow);
        mem_shadow->set_main_dmem((DataMem*)mem);
        mem_shadow->set_is_shadow(true);
        ((DataMem*)mem)->set_has_shadow(true);
        //current_kernel_->replace_with_shadow_dmem(mem_shadow, current_idx_, current_param_info_);
        Task* current_writing_task = mem->get_current_writing_task(); 

        create_shadow_flush(cmd, current_writing_task, mem); // flushes to shadow host pointer

        //std::cout << "Current Task "<< task->name() << "Current Writing Task " << current_writing_task->name() << std::endl;
	    //Task* task_shadow_flush = mem->get_flush_task();
        task->AddDepend(mem->get_flush_task(), mem->get_flush_task()->uid());
        //task->AddDepend(mem->get_flush_task());	
        //printf("For %s -> parent %s auto shadow\n", task->name(), mem->get_flush_task()->name());
        //task_shadow_flush->AddDepend(current_writing_task);	
        

    }
}

void AutoDAG::create_shadow_flush(Command* cmd, Task* task, BaseMem* mem) {
	Task* task_flush = mem->get_flush_task();
	if ( task_flush == NULL)
        	_error("Flush Task is NULL which is not possible for shadow case:%ld:%s\n",
                    task->uid(), task->name());

    sprintf(tn, "%s-to-shadow-flushed-out", task->name());
    //printf("For %s Before Shadow flush task name %s\n", task->name(), task_flush->name());
    task_flush->set_name(tn);
    //printf("For %s After Shadow flush task name %s\n", task->name(), task_flush->name());
    task_flush->ClearCommands();
    Command* cmd_shadow_flush = Command::CreateMemFlushOutToShadow(task_flush, (DataMem *) mem);
    task_flush->AddCommand(cmd_shadow_flush);
}
#endif

#ifdef AUTO_FLUSH
void AutoDAG::create_auto_flush(Command* cmd, Task* task, 
		BaseMem* mem) {

#ifndef AUTO_SHADOW
	Task* task_flush = mem->get_flush_task();
#else
    if(((DataMem*)mem)->get_has_shadow() == true) {
        mem = ((BaseMem*)((DataMem*)mem)->get_current_dmem_shadow());
    }
	Task* task_flush = mem->get_flush_task();
#endif
	Graph* graph_flush = current_graph_;
	if(graph_flush == NULL) return;
	//Graph* graph_flush = cmd->platform_->get_current_graph();
	//printf("-------Flush task Name: %s----------\n", task_flush->name());
    //char tn[256];
#ifndef AUTO_SHADOW
    sprintf(tn, "%s-auto-flushed-out", task->name());
#else
    if(((DataMem*)mem)->get_is_shadow() == false) 
        sprintf(tn, "%s-auto-flushed-out", task->name());
    else
        sprintf(tn, "%s-from-shadow-flushed-out", task->name());
#endif


#ifdef INCORRECT_AUTO_FLUSH
	if ( task_flush == NULL || task_flush != NULL ){
#else
	if ( task_flush == NULL){
#endif
	    task_flush = Task::Create(platform_, IRIS_TASK, tn);
	    //task_flush = Task::Create(cmd->platform_, IRIS_TASK, tn);
        Command* cmd_flush;

#ifndef AUTO_SHADOW
        cmd_flush = Command::CreateMemFlushOut(task_flush, (DataMem *) mem);
#else
        if(((DataMem*)mem)->get_is_shadow() == false) 
            cmd_flush = Command::CreateMemFlushOut(task_flush, (DataMem *) mem);
        else
            cmd_flush = Command::CreateMemFlushOutToShadow(task_flush, (DataMem *) mem);
#endif
        task_flush->AddCommand(cmd_flush);
   	    mem->set_flush_task(task_flush);
        //task_flush->AddDepend(task);	
        task_flush->AddDepend(task, task->uid());
        //printf("For %s -> parent %s auto flush\n", task_flush->name(), task->name());
	    if(graph_flush != NULL) {
		    //graph_flush->AddTask(task_flush);
            graph_flush->AddTask(task_flush, task_flush->uid());
	    }else {
		//printf("-------Graph NULL----------\n");
        	_error("Graph is NULL:%ld:%s\n", task->uid(), task->name());
	    }

#ifdef INCORRECT_AUTO_FLUSH
 	    mem->set_current_writing_task(task_flush);
#endif

	} else {
	    //printf("-------Flush task Null----------\n");
    	//printf("Total dependency %d\n", task_flush->ndepends());
        //printf("For %s Before Auto flush task name %s\n", task->name(), task_flush->name());
	    task_flush->set_name(tn);
        //printf("For %s After Auto flush task name %s\n", task->name(), task_flush->name());
        task_flush->ReplaceDependFlushTask(task);	
        //task_flush->EraseDepend();
	    //printf("Total dependency %d\n",task_flush->ndepends());
	}

#ifdef AUTO_SHADOW
    if(((DataMem*)mem)->get_is_shadow() == true) {
        //if(task_flush->get_shadow_dep_added() == false){
	        create_multi_read_dependency(task_flush, ((DataMem*)mem)->get_main_dmem());
            //task_flush->set_shadow_dep_added(true);
        //}
    }
#endif

	//task_flush->set_opt(task->get_opt());
	//task_flush->set_brs_policy(task->brs_policy());
    //task_flush->set_df_scheduling();

	//task_flush->set_brs_policy(task->get_brs_policy());
}
#endif

#ifdef SANITY_CHECK

void AutoDAG::extra_dependencies() {
    std::cout << "---List of Extra dependencies---" << std::endl;
    for(int i = 0 ; i < manual_dep_list_.size(); i++) {
        int found = 0;
        for(int j = 0 ; j < auto_dep_list_.size(); j++) {
            if ( manual_dep_list_[i] == auto_dep_list_[j]) found = 1;
        }
        if (found == 0) std::cout << "   " << manual_dep_list_[i] << std::endl;
    }
    std::cout << "\n\n\n";
}

void AutoDAG::missing_dependencies() {
    std::cout << "---List of Missing dependencies---" << std::endl;
    for(int i = 0 ; i < auto_dep_list_.size(); i++) {
        int found = 0;
        for(int j = 0 ; j < manual_dep_list_.size(); j++) {
            if ( auto_dep_list_[i] == manual_dep_list_[j]) found = 1;
        }
        if (found == 0) std::cout << "   " << auto_dep_list_[i] << std::endl;
    }
    std::cout << "\n\n\n";
}
#endif



int iris_bc_mapping(int row, int col)
{
    int ndevices = 4;
    int dev_map[16][16];
    for(int i=0; i<16; i++) {
        for(int j=0; j<16; j++) {
            if (j>=i) dev_map[i][j] = j*j+i;
            else dev_map[i][j] = (i*(i+2))-j;
            dev_map[i][j] = dev_map[i][j]%ndevices;
        }
    }
    int nrows = ndevices;
    int ncols = ndevices;
    if (ndevices == 1) return 0;
    else if (ndevices == 9) {
        nrows = 3; ncols = 3;
    }
    else if (ndevices == 4) {
        nrows = 2; ncols = 2;
    }
    else if (ndevices == 6) {
        dev_map[0][0] = 0;
        dev_map[0][1] = 1;
        dev_map[1][0] = 2;
        dev_map[1][1] = 3;
        dev_map[2][0] = 4;
        dev_map[2][1] = 5;
        nrows = 3; ncols = 2;
    }
    else if (ndevices == 8) {
        dev_map[0][0] = 0;
        dev_map[0][1] = 1;
        dev_map[1][0] = 2;
        dev_map[1][1] = 3;
        dev_map[2][0] = 4;
        dev_map[2][1] = 5;
        dev_map[3][0] = 6;
        dev_map[3][1] = 7;
        nrows = 4; ncols = 2;
    }
    else if (ndevices % 2 == 1) {
        int incrementer = (ndevices+1) / 2;
        int i_pos = 0;
        for(int i=0; i<ndevices; i++) {
            int j_pos = (ndevices - i_pos)%ndevices;
            for(int j=0; j<ndevices; j++) {
                dev_map[i][j] = (j_pos + j)%ndevices;
            }
            i_pos = (i_pos+incrementer)%ndevices;
        }
    }
    /*printf("Dev Map:\n");
    for(int i=0; i<nrows; i++) {
        for(int j=0; j<ncols; j++) {
            printf("%2d ", dev_map[i][j]);
        }
        printf("\n");
    }*/
    return dev_map[row%nrows][col%ncols];
}

void AutoDAG::df_bc_scheduling(Task *task, DataMem* mem)
{
    /*printf("Task %s, uid %d Row %d, Col %d, bc %d, num device %d\n", task->name(), 
        mem->uid(), mem->get_row(), 
        mem->get_col(), mem->get_bc(), num_dev_);*/
    int dmem_dev = -1;


    if(mem->get_rr_bc_dev() == -1) {
        
        dmem_dev = cur_dev_++;
        mem->set_rr_bc_dev(dmem_dev);
        if (cur_dev_ == num_dev_) cur_dev_ = 0;
        
       
       /*dmem_dev = (mem->get_col() + mem->get_row())% num_dev_;
       mem->set_rr_bc_dev(dmem_dev); */
       
       //dmem_dev = iris_bc_mapping(mem->get_row(), mem->get_col());
       //mem->set_rr_bc_dev(dmem_dev);

    } else {
        dmem_dev = mem->get_rr_bc_dev();
    }
    task->set_brs_policy(dmem_dev);
    task->set_df_scheduling();
}


void AutoDAG::add_h2d_df_task(Task* task, Kernel* kernel){
    /*
    DataMem* dmem;
    if(task->get_df_scheduling() != true) {
        printf("no target is set for task %s\n", task->name());
        return;
    }
    int task_dev = task->get_brs_policy(); 
    Task* df_task = h2d_task_list_.at(task_dev);
    for (size_t i = 0; i < kernel->all_data_mems_in().size(); i++){
        dmem = (DataMem*) kernel->all_data_mems_in().at(i);
        if (dmem->get_h2d_df_flag(task_dev) == false){
            Command* cmd_h2d = Command* Command::CreateH2D(task, mem, size_t *off, size_t *host_sizes, size_t *dev_sizes, size_t elem_size, int dim, void* host) {
            df_task->AddCommand(cmd_h2d);

            dmem->set_h2d_df_flag(task_dev) 
        }

    } */
}

} /* namespace rt */
} /* namespace iris */


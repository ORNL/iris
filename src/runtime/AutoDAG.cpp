#include "AutoDAG.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "Debug.h"
#include "Platform.h"
#include "Kernel.h"
#include <stdlib.h>

#include <iostream>

namespace iris {
namespace rt {

AutoDAG::AutoDAG(Platform* platform){ 
    platform_ = platform;
#ifdef AUTO_FLUSH
    current_graph_ = NULL;
#endif
}

//AutoDAG::platform_ = NULL;

void AutoDAG::create_dependency(Command* cmd, Task* task, 
		int param_info, BaseMem* mem, 
        Kernel* kernel, int idx) {
    current_kernel_ = kernel;
    current_idx_ = idx;
    current_param_info_ = param_info;

#ifdef AUTO_SHADOW

    if( mem->GetMemHandlerType() == IRIS_DMEM) {
        if(((DataMem*)mem)->get_is_shadow() == true) printf("----------------no ghapla---------------\n");
        if(((DataMem*)mem)->get_has_shadow() == true) printf("----------------ghapla---------------\n");
    }
#endif
    //printf(" Task %s param_info %d\n", task->name(), param_info);
    //printf(" Task %s , mem read count %d\n", task->name(), mem->get_read_task_list()->size());
    //Task* task_prev = NULL;
    if (param_info == iris_w || param_info == iris_rw) {
    	if(mem->get_current_writing_task() != NULL 
		    && mem->get_read_task_list()->size() == 0 ){
    	    if (mem->get_current_writing_task() != task) {
           	    task->AddDepend(mem->get_current_writing_task());	
                printf("For %s -> parent %s main w or wr\n", task->name(), mem->get_current_writing_task()->name());
	        }
    	} else if (  mem->get_read_task_list()->size() != 0 ){
#ifndef AUTO_SHADOW
	        create_multi_read_dependency(task, mem);
#else
            if( mem->GetMemHandlerType() == IRIS_DMEM)
	            create_auto_shadow(cmd, task, mem);
            else
	            create_multi_read_dependency(task, mem);
#endif 
 	        mem->erase_all_read_task_list();
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
                task->AddDepend(mem->get_current_writing_task());	
                //printf("For %s -> parent %s main r shadow: %d\n", task->name(), mem->get_current_writing_task()->name(), ((DataMem*)mem)->get_is_shadow() );
	        }
	    }
  	    task->add_to_read_list(mem);
	    if (mem->get_current_writing_task() != task) {
  	        mem->add_to_read_task_list(task);
  	        //set_current_writing_task(NULL);
	    }
    }

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
            task->AddDepend(read_task);	
            printf("For %s -> parent %s multiread\n", task->name(), read_task->name());
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
    if( read_list->size() < 5 || ((DataMem*)mem)->get_is_shadow() == true) {
	    create_multi_read_dependency(task, mem);
    } else {
        std::cout << "For " << task->name() << " shadow creation " << std::endl;
        //std::cout << "Host size  " << *(((DataMem*)mem)->host_size()) << std::endl;
        //std::cout << "device size  " << ((DataMem*)mem)->size() << std::endl;
        void* host_ptr_shadow;
        //host_ptr_shadow = malloc(((DataMem*)mem)->size()); // commented out because we dont need
        //((DataMem*)mem)->set_host_ptr_shadow(host_ptr_shadow);
        
        DataMem* mem_shadow = new DataMem(platform_, 
                host_ptr_shadow, mem->size()); // TODO try with NULL
        platform_->insert_into_mems(mem_shadow);

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
        task->AddDepend(mem->get_flush_task());	
        printf("For %s -> parent %s auto shadow\n", task->name(), mem->get_flush_task()->name());
        //task_shadow_flush->AddDepend(current_writing_task);	
        

    }
}

void AutoDAG::create_shadow_flush(Command* cmd, Task* task, BaseMem* mem) {
	Task* task_flush = mem->get_flush_task();
	if ( task_flush == NULL)
        	_error("Flush Task is NULL which is not possible for shadow case:%ld:%s\n",
                    task->uid(), task->name());

    sprintf(tn, "%s-to-shadow-flushed-out", task->name());
    printf("For %s Before Shadow flush task name %s\n", task->name(), task_flush->name());
    task_flush->set_name(tn);
    printf("For %s After Shadow flush task name %s\n", task->name(), task_flush->name());
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
    if(((DataMem*)mem)->get_has_shadow() == true)
        mem =((BaseMem*)((DataMem*)mem)->get_current_dmem_shadow());
	Task* task_flush = mem->get_flush_task();
#endif
	Graph* graph_flush = current_graph_;
	if(graph_flush == NULL) return;
	//Graph* graph_flush = cmd->platform_->get_current_graph();
	//printf("-------Flush task Name: %s----------\n", task_flush->name());
    //char tn[256];
#ifdef AUTO_SHADOW
    if(((DataMem*)mem)->get_is_shadow() == false) 
#endif
        sprintf(tn, "%s-auto-flushed-out", task->name());
#ifdef AUTO_SHADOW
    else
        sprintf(tn, "%s-from-shadow-flushed-out", task->name());
#endif

	if ( task_flush == NULL){
	    task_flush = Task::Create(platform_, IRIS_TASK, tn);
	    //task_flush = Task::Create(cmd->platform_, IRIS_TASK, tn);
        Command* cmd_flush;

#ifdef AUTO_SHADOW
        if(((DataMem*)mem)->get_is_shadow() == false) 
#endif
            cmd_flush = Command::CreateMemFlushOut(task_flush, (DataMem *) mem);
#ifdef AUTO_SHADOW
        else
            cmd_flush = Command::CreateMemFlushOutToShadow(task_flush, (DataMem *) mem);
#endif
        task_flush->AddCommand(cmd_flush);
   	    mem->set_flush_task(task_flush);
        task_flush->AddDepend(task);	
        printf("For %s -> parent %s auto flush\n", task_flush->name(), task->name());
	    if(graph_flush != NULL) {
		    graph_flush->AddTask(task_flush);
	    }else {
		printf("-------Graph NULL----------\n");
        	_error("Graph is NULL:%ld:%s\n", task->uid(), task->name());
	    }
	} else {
	    //printf("-------Flush task Null----------\n");
    	//printf("Total dependency %d\n", task_flush->ndepends());
        printf("For %s Before Auto flush task name %s\n", task->name(), task_flush->name());
	    task_flush->set_name(tn);
        printf("For %s After Auto flush task name %s\n", task->name(), task_flush->name());
        task_flush->ReplaceDependFlushTask(task);	
        //task_flush->EraseDepend();
	    //printf("Total dependency %d\n",task_flush->ndepends());
	}
	task_flush->set_opt(task->get_opt());
	task_flush->set_brs_policy(task->get_brs_policy());
}
#endif

} /* namespace rt */
} /* namespace iris */


#include "AutoDAG.h"
#include "BaseMem.h"
#include "DataMem.h"
#include "Debug.h"
#include "Platform.h"
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
		int param_info, 
		BaseMem* mem) {
    printf(" Task %s , mem read count %d\n", 
        task->name(), mem->get_read_task_list()->size());
    Task* task_prev = NULL;
    if (param_info == iris_w || param_info == iris_rw) {
    	if(mem->get_current_writing_task() != NULL 
		    && mem->get_read_task_list()->size() == 0 ){
    	    if (mem->get_current_writing_task() != task) {
           	    task->AddDepend(mem->get_current_writing_task());	
	        }
    	} else if (  mem->get_read_task_list()->size() != 0 ){
#ifndef AUTO_SHADOW
	        create_multi_read_dependency(task, mem);
#else
	        create_auto_shadow(cmd, task, mem);
#endif 
        }
 
    //printf("Seting current task info %d\n", param_info);
	task_prev = mem->get_current_writing_task();
 	mem->set_current_writing_task(task);
 	mem->erase_all_read_task_list();
  	task->add_to_write_list(mem);

#ifdef AUTO_FLUSH
 	create_auto_flush(cmd, task, mem);
#endif
    }
    if (param_info == iris_r) {
    	if(mem->get_current_writing_task() != NULL) {
    	    if (mem->get_current_writing_task() != task) {
                task->AddDepend(mem->get_current_writing_task());	
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
	    }
    }
}

#ifdef AUTO_SHADOW
void AutoDAG::create_auto_shadow(Command* cmd, Task* task, 
		BaseMem* mem) {
    std::vector<Task*>* read_list =  
	    mem->get_read_task_list();
    if( read_list->size() < 2 ) {
	    create_multi_read_dependency(task, mem);
    } else {
        std::cout << " Current Task " << task->name() << std::endl;
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

        create_shadow_flush(cmd, task, mem); // flushes to shadow host pointer

        Task* current_writing_task = mem->get_current_writing_task(); 
        std::cout << " Current Writing Task " << current_writing_task->name() << std::endl;
	    /*Task* task_shadow_flush = mem->get_flush_task();
        task->AddDepend(task_shadow_flush);	
 	    create_auto_flush(cmd, task, mem);*/
    }
}

void AutoDAG::create_shadow_flush(Command* cmd, Task* task, BaseMem* mem) {
	Task* task_flush = mem->get_flush_task();
	if ( task_flush == NULL)
        	_error("Flush Task is NULL which is not possible for shadow case:%ld:%s\n",
                    task->uid(), task->name());

    sprintf(tn, "%s-shadow-flushed-out", task->name());
    task_flush->set_name(tn);
    task_flush->ClearCommands();
    //Command* cmd_shadow_flush = 
        //Command::CreateMemFlushOutToShadow(task_flush, (DataMem *) mem);
    Command* cmd_shadow_flush = Command::CreateMemFlushOutToShadow(task_flush, (DataMem *) mem);
    task_flush->AddCommand(cmd_shadow_flush);
}
#endif

#ifdef AUTO_FLUSH
void AutoDAG::create_auto_flush(Command* cmd, Task* task, 
		BaseMem* mem) {

	Task* task_flush = mem->get_flush_task();
	Graph* graph_flush = current_graph_;
	if(graph_flush == NULL) return;
	//Graph* graph_flush = cmd->platform_->get_current_graph();
	//printf("-------Flush task Name: %s----------\n", task_flush->name());
    //char tn[256];
    sprintf(tn, "%s-auto-flushed-out", task->name());
	if ( task_flush == NULL){
	    task_flush = Task::Create(platform_, IRIS_TASK, tn);
	    //task_flush = Task::Create(cmd->platform_, IRIS_TASK, tn);
        Command* cmd_flush = Command::CreateMemFlushOut(task_flush, (DataMem *) mem);
        task_flush->AddCommand(cmd_flush);
   	    mem->set_flush_task(task_flush);
        task_flush->AddDepend(task);	
	    if(graph_flush != NULL) {
		    graph_flush->AddTask(task_flush);
	    }else {
		printf("-------Graph NULL----------\n");
        	_error("Graph is NULL:%ld:%s\n", task->uid(), task->name());
	    }
	} else {
	    //printf("-------Flush task Null----------\n");
    	//printf("Total dependency %d\n", task_flush->ndepends());
	    task_flush->set_name(tn);
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


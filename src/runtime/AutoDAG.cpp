#include "AutoDAG.h"
#include "BaseMem.h"
#include "Debug.h"
#include "Platform.h"
#include <stdlib.h>


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
		BaseMem* mem, Task* task_prev) {

    if (param_info == iris_w || param_info == iris_rw) {

    	if(mem->get_current_writing_task() != NULL 
		    && mem->get_read_task_list()->size() == 0 ){
    	    if (mem->get_current_writing_task() != task) {
            	    task->AddDepend(mem->get_current_writing_task());	
	    }
    	} else if (  mem->get_read_task_list()->size() != 0 ){
  	    std::vector<Task*>* read_list =  mem->get_read_task_list();
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
 
	//printf("Seting current task info %d\n", param_info);
	task_prev = mem->get_current_writing_task();
 	mem->set_current_writing_task(task);
 	mem->erase_all_read_task_list();
  	task->add_to_write_list(mem);

#ifdef AUTO_FLUSH
 	create_auto_flush(cmd, task, param_info, mem, task_prev);
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

#ifdef AUTO_FLUSH
void AutoDAG::create_auto_flush(Command* cmd, Task* task, 
		int param_info, 
		BaseMem* mem, Task* task_prev) {

    	//sprintf(tn, "%s-flush-out", mem->get_flush_task();
	//cmd->platform_->TaskCreate(tn, false, mem->get_flush_task())
	Task* task_flush = mem->get_flush_task();
	Graph* graph_flush = current_graph_;
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
	
  	//platform_->GraphTask(get_graph(), mem->get_flush_task(), 1, NULL);
        //platform_->TaskMemFlushOut(mem->get_flush_task(), mem) {
        //task->AddDepend(mem->get_flush_task());	
    	//iris_graph_task( graph, mem->get_flush_task(), 1, NULL );
  	//platform_->GraphTask(get_graph(), mem->get_flush_task(), 1, NULL);
    	//mem->get_flush_task(
}
#endif

} /* namespace rt */
} /* namespace iris */


extern crate libc;
include!(concat!(env!("IRIS"), "/include/iris/iris.rs"));
//use libc::{c_int, c_void, size_t, malloc, free};
use std::ptr::{null_mut, null};
use std::ffi::CString;

struct IrisStruct {
    mem: iris_mem,
}

impl IrisStruct {
    fn new(size: usize, host_ptr: *mut c_void) -> Result<Self, String> {
        let mut mem = null_mut();
        let res = unsafe { iris_data_mem_create(&mut mem, host_ptr, size) };
        if res != 0 {
            Err(format!("Failed to create memory object: {}", res))
        } else {
            Ok(IrisStruct { mem })
        }
    }

    fn release(&self) {
        unsafe {
            iris_mem_release(self.mem);
        }
    }
}

fn main() {

    let mut argc = std::env::args().len() as c_int;
    let argv: Vec<CString> = std::env::args().map(|arg| CString::new(arg).unwrap()).collect();
    let argv_ptrs: Vec<*mut c_char> = argv.iter().map(|arg| arg.as_ptr() as *mut c_char).collect();
    let mut argv_ptr = argv_ptrs.as_ptr() as *mut *mut c_char;
    let A: f32 = 10.0; // Explicitly specifying the type as f32
    let ERROR: i32 = 0;
    unsafe {
        iris_init(&mut argc, &mut argv_ptr, 1);
    }

    let size = std::env::args().nth(1).map_or(8, |v| v.parse::<usize>().unwrap());
    let target = std::env::args().nth(2).map_or(0, |v| v.parse::<c_int>().unwrap());
    let verbose = std::env::args().nth(3).map_or(1, |v| v.parse::<c_int>().unwrap());

    let x = unsafe { malloc(size * std::mem::size_of::<f32>()) as *mut f32 };
    let y = unsafe { malloc(size * std::mem::size_of::<f32>()) as *mut f32 };
    let z = unsafe { malloc(size * std::mem::size_of::<f32>()) as *mut f32 };

    if !x.is_null() && !y.is_null() && !z.is_null() {
        for i in 0..size {
            unsafe {
                *x.add(i) = i as f32;
                *y.add(i) = i as f32;
            }
        }

        if verbose != 0 {
            println!("X [");
            for i in 0..size {
                unsafe {
                    print!(" {:.0}.", *x.add(i));
                }
            }
            println!("]\nY [");
            for i in 0..size {
                unsafe {
                    print!(" {:.0}.", *y.add(i));
                }
            }
            println!("]");
        }

        let mem_x = IrisStruct::new(size * std::mem::size_of::<f32>(), x as *const c_void).unwrap();
        let mem_y = IrisStruct::new(size * std::mem::size_of::<f32>(), y as *const c_void).unwrap();
        let mem_z = IrisStruct::new(size * std::mem::size_of::<f32>(), z as *const c_void).unwrap();

        let mut task0 = null_mut();
        unsafe {
            iris_task_create(&mut task0);
            let kernel_name = CString::new("saxpy").unwrap();
            let saxpy_params: [*const c_void; 4] = [&mem_z.mem as *const _ as *const c_void, &A as *const _ as *const c_void, &mem_x.mem as *const _ as *const c_void, &mem_y.mem as *const _ as *const c_void];
            let saxpy_params_info: [c_int; 4] = [iris_w, std::mem::size_of::<f32>() as c_int, iris_r, iris_r];
            iris_task_kernel(task0, kernel_name.as_ptr(), 1, null(), &size as *const _, null(), 4, saxpy_params.as_ptr(), saxpy_params_info.as_ptr());
            iris_task_submit(task0, target, null(), 1);
            iris_task_dmem_flush_out(task0, mem_z.mem);
        }

        if verbose != 0 {
            for i in 0..size {
                unsafe {
                    if *z.add(i) != A * *x.add(i) + *y.add(i) {
                        ERROR += 1;
                    }
                }
            }

            println!("S = {} * X + Y [", A);
            for i in 0..size {
                unsafe {
                    print!(" {:.0}.", *z.add(i));
                }
            }
            println!("]");
        }

        mem_x.release();
        mem_y.release();
        mem_z.release();
    }

    unsafe {
        free(x as *mut c_void);
        free(y as *mut c_void);
        free(z as *mut c_void);
        iris_finalize();
    }
}


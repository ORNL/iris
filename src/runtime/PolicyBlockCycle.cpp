#include "PolicyBlockCycle.h"
#include "Debug.h"
#include "Task.h"
#include "Scheduler.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

namespace iris {
    namespace rt {

        class DeviceMap {
            private:
                int dev_map[16][16];
                int nrows, ncols;
                int ndevices;
                // Private constructor to prevent instantiation outside the class
                DeviceMap() {
                    ndevices = 0;
                    iris_device_count(&ndevices);
                    for(int i=0; i<16; i++) {
                        for(int j=0; j<16; j++) {
                            if (j>=i) dev_map[i][j] = j*j+i;
                            else dev_map[i][j] = (i*(i+2))-j;
                            dev_map[i][j] = dev_map[i][j]%ndevices;
                        }
                    }
                    nrows = ndevices;
                    ncols = ndevices;
                    if (ndevices == 1) return;
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

                }
                // Deleting copy constructor and assignment operator
                DeviceMap(const DeviceMap&) = delete;
                DeviceMap& operator=(const DeviceMap&) = delete;

            public:
                int GetDevice(int r, int c) 
                {
                    if (r>=0 && c>=0) {
                        int id = dev_map[r%nrows][c%ncols];
                        return id;
                    }
                    else {
                        return -1;
                    }
                }

                // Public method to access the instance of the class
                static DeviceMap& getInstance() {
                    static DeviceMap instance;
                    return instance;
                }
        };
        PolicyBlockCycle::PolicyBlockCycle(Scheduler* scheduler) {
            SetScheduler(scheduler);
        }

        PolicyBlockCycle::~PolicyBlockCycle() {
        }

        void PolicyBlockCycle::GetDevices(Task* task, Device** devs, int* ndevs) 
        {
            DeviceMap & dev_map = DeviceMap::getInstance();
            int r = task->metadata(0);
            int c = task->metadata(1);
            int id = dev_map.GetDevice(r, c);
            int n = 0;
            if (id >= 0) {
                devs[n++] = devs_[id];
                *ndevs = n;
            }
            else {
                for (int i = 0; i < ndevs_; i++) 
                    if (IsKernelSupported(task, devs_[i]))
                        devs[n++] = devs_[i];
                *ndevs = n;
            }
        }

    } /* namespace rt */
} /* namespace iris */



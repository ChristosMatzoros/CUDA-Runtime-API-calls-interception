/*********************

Matzoros Christos , UTH , 1/5/2020

The purpose of this code is to intercept CUDA Runtime API calls and to handle them accordingly.

***********************/


//Headers
#include <stdio.h>
#include <list>
#include <map>
#include <cassert>
#include <vector_types.h>
#include <dlfcn.h>  //for dynamic linking
#include <cuda.h>
#include <driver_types.h>
using namespace std;



typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    list <void*> arguments;
    int counter;
} kernel_info_t;

static list<kernel_info_t> kernels_list;

kernel_info_t &kernelInfo() {
    static kernel_info_t kernelInfo;
    return kernelInfo;
}



/////////////////////////
//   PRINT FUNCTIONS   //
/////////////////////////

void print_grid_dimensions(dim3 gridDim){
    if (gridDim.y == 1 && gridDim.z == 1) {     //1D grid (x)
        printf("gridDim=%d ", gridDim.x);
    } else if (gridDim.z == 1) {    //2D grid (x,y)
        printf("gridDim=[%d,%d] ", gridDim.x, gridDim.y);
    } else { //3D grid (x,y,z)
        printf("gridDim=[%d,%d,%d] ", gridDim.x, gridDim.y, gridDim.z);
    }
}

void print_block_dimensions(dim3 blockDim){
    if (blockDim.y == 1 && blockDim.z == 1) {   //1D block (x)
        printf("blockDim=%d ", blockDim.x);
    } else if (blockDim.z == 1) {   //2D block (x,y)
        printf("blockDim=[%d,%d] ", blockDim.x, blockDim.y);
    } else {    //3D block (x,y,z)
        printf("blockDim=[%d,%d,%d] ", blockDim.x, blockDim.y, blockDim.z);
    }
}

void print_dimensions(dim3 gridDim, dim3 blockDim){
    print_grid_dimensions(gridDim);
    print_block_dimensions(blockDim);
}

void print_args(list <void*> arg){
    for (std::list<void *>::iterator it = arg.begin(), end = arg.end(); it != end; ++it) {
        unsigned i = std::distance(arg.begin(), it);
        printf("%d:%d \n", i, *(static_cast<int *>(*it)));
    }
}

void print_kernel_invocation(const char *entry) {
    printf("New kernel invocation\n");
    print_dimensions(kernelInfo().gridDim,kernelInfo().blockDim);
    //print_args(kernelInfo().arguments);
    printf("\n");
}



////////////////////////////
//   CALLS INTERCEPTION   //
////////////////////////////

//*******************************************//
//      CUDA Runtime API Error Handling      //
//*******************************************//
///   cudaGetErrorName   ///
typedef const char* (*cudaGetErrorName_t)(cudaError_t error);
static cudaGetErrorName_t native_cudaGetErrorName = NULL;

extern "C" const char* cudaGetErrorName(cudaError_t error) {
    printf("\n>> cudaGetErrorName interception\n");

    if (native_cudaGetErrorName == NULL) {
        native_cudaGetErrorName = (cudaGetErrorName_t)dlsym(RTLD_NEXT,"cudaGetErrorName");
    }
    assert(native_cudaGetErrorName != NULL);
    return native_cudaGetErrorName(error);
}

///   cudaGetErrorString   ///
typedef const char* (*cudaGetErrorString_t)(cudaError_t error);
static cudaGetErrorString_t native_cudaGetErrorString = NULL;

extern "C" const char* cudaGetErrorString(cudaError_t error) {
    printf("\n>> cudaGetErrorString interception\n");

    if (native_cudaGetErrorString == NULL) {
        native_cudaGetErrorString = (cudaGetErrorString_t)dlsym(RTLD_NEXT,"cudaGetErrorString");
    }
    assert(native_cudaGetErrorString != NULL);
    return native_cudaGetErrorString(error);
}

///   cudaGetLastError   ///
typedef cudaError_t (*cudaGetLastError_t)(void);
static cudaGetLastError_t native_cudaGetLastError = NULL;

extern "C" cudaError_t cudaGetLastError(void) {
    printf("\n>> cudaGetLastError interception\n");

    if (native_cudaGetLastError == NULL) {
        native_cudaGetLastError = (cudaGetLastError_t)dlsym(RTLD_NEXT,"cudaGetLastError");
    }
    assert(native_cudaGetLastError != NULL);
    return native_cudaGetLastError();
}

///   cudaGetLastError   ///
typedef cudaError_t (*cudaPeekAtLastError_t)(void);
static cudaPeekAtLastError_t native_cudaPeekAtLastError = NULL;

extern "C" cudaError_t cudaPeekAtLastError(void) {
    printf("\n>> cudaPeekAtLastError interception\n");

    if (native_cudaPeekAtLastError== NULL) {
        native_cudaPeekAtLastError = (cudaPeekAtLastError_t)dlsym(RTLD_NEXT,"cudaPeekAtLastError");
    }
    assert(native_cudaPeekAtLastError != NULL);
    return native_cudaPeekAtLastError();
}


//**********************************************//
//      CUDA Runtime API Device Management      //
//**********************************************//
///   cudaChooseDevice   ///
typedef cudaError_t (*cudaChooseDevice_t)(int * device, const struct cudaDeviceProp * prop);
static cudaChooseDevice_t native_cudaChooseDevice = NULL;

extern "C" cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop) {
    printf("\n>>cudaChooseDevice interception \n");

    if (native_cudaChooseDevice == NULL) {
        native_cudaChooseDevice = (cudaChooseDevice_t)dlsym(RTLD_NEXT,"cudaChooseDevice");
    }
    assert(native_cudaChooseDevice != NULL);
    return native_cudaChooseDevice(device,prop);
}

///   cudaDeviceGetAttribute   ///
typedef cudaError_t (*cudaDeviceGetAttribute_t)(int* value, cudaDeviceAttr attr, int device);
static cudaDeviceGetAttribute_t native_cudaDeviceGetAttribute = NULL;

extern "C" cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    printf("\n>>cudaDeviceGetAttribute interception \n");

    if (native_cudaDeviceGetAttribute == NULL) {
        native_cudaDeviceGetAttribute = (cudaDeviceGetAttribute_t)dlsym(RTLD_NEXT,"cudaDeviceGetAttribute");
    }
    assert(native_cudaDeviceGetAttribute != NULL);
    return native_cudaDeviceGetAttribute(value,attr,device);
}

///   cudaDeviceGetByPCIBusId    ///
typedef cudaError_t (*cudaDeviceGetByPCIBusId_t)(int* device, const char* pciBusId);
static cudaDeviceGetByPCIBusId_t native_cudaDeviceGetByPCIBusId  = NULL;

extern "C" cudaError_t cudaDeviceGetByPCIBusId  (int* device, const char* pciBusId) {
    printf("\n>>cudaDeviceGetByPCIBusId  interception\n");

    if (native_cudaDeviceGetByPCIBusId  == NULL) {
        native_cudaDeviceGetByPCIBusId  = (cudaDeviceGetByPCIBusId_t)dlsym(RTLD_NEXT,"cudaDeviceGetByPCIBusId ");
    }
    assert(native_cudaDeviceGetByPCIBusId  != NULL);
    return native_cudaDeviceGetByPCIBusId (device,pciBusId);
}

///   cudaDeviceGetCacheConfig   ///
typedef cudaError_t (*cudaDeviceGetCacheConfig_t)(cudaFuncCache ** pCacheConfig);
static cudaDeviceGetCacheConfig_t native_cudaDeviceGetCacheConfig = NULL;

extern "C" cudaError_t cudaDeviceGetCacheConfig (cudaFuncCache ** pCacheConfig) {
    printf("\n>>cudaDeviceGetCacheConfig interception\n");

    if (native_cudaDeviceGetCacheConfig == NULL) {
        native_cudaDeviceGetCacheConfig = (cudaDeviceGetCacheConfig_t)dlsym(RTLD_NEXT,"cudaDeviceGetCacheConfig");
    }
    assert(native_cudaDeviceGetCacheConfig != NULL);
    return native_cudaDeviceGetCacheConfig(pCacheConfig);
}

///   cudaDeviceGetLimit   ///
typedef cudaError_t (*cudaDeviceGetLimit_t)(size_t* pValue, cudaLimit limit);
static cudaDeviceGetLimit_t native_cudaDeviceGetLimit = NULL;

extern "C" cudaError_t cudaDeviceGetLimit (size_t* pValue, cudaLimit limit) {
    printf("\n>>cudaDeviceGetLimit interception\n");

    if (native_cudaDeviceGetLimit == NULL) {
        native_cudaDeviceGetLimit = (cudaDeviceGetLimit_t)dlsym(RTLD_NEXT,"cudaDeviceGetLimit");
    }
    assert(native_cudaDeviceGetLimit != NULL);
    return native_cudaDeviceGetLimit(pValue,limit);
}

///   cudaDeviceGetNvSciSyncAttributes   ///
typedef cudaError_t (*cudaDeviceGetNvSciSyncAttributes_t)( void* nvSciSyncAttrList, int device, int flags);
static cudaDeviceGetNvSciSyncAttributes_t native_cudaDeviceGetNvSciSyncAttributes = NULL;

extern "C" cudaError_t cudaDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, int device, int flags) {
    printf("\n>>cudaDeviceGetNvSciSyncAttributes interception\n");

    if (native_cudaDeviceGetNvSciSyncAttributes== NULL) {
        native_cudaDeviceGetNvSciSyncAttributes= (cudaDeviceGetNvSciSyncAttributes_t)dlsym(RTLD_NEXT,"cudaDeviceGetNvSciSyncAttributes");
    }
    assert(native_cudaDeviceGetNvSciSyncAttributes != NULL);
    return native_cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList,device,flags);
}

///   cudaDeviceGetP2PAttribute   ///
typedef cudaError_t (*cudaDeviceGetP2PAttribute_t)(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
static cudaDeviceGetP2PAttribute_t native_cudaDeviceGetP2PAttribute= NULL;

extern "C" cudaError_t cudaDeviceGetP2PAttribute (int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
    printf("\n>>cudaDeviceGetP2PAttribute interception\n");

    if (native_cudaDeviceGetP2PAttribute == NULL) {
        native_cudaDeviceGetP2PAttribute = (cudaDeviceGetP2PAttribute_t)dlsym(RTLD_NEXT,"cudaDeviceGetP2PAttribute");
    }
    assert(native_cudaDeviceGetP2PAttribute != NULL);
    return native_cudaDeviceGetP2PAttribute(value,attr,srcDevice,dstDevice);
}

///   cudaDeviceGetPCIBusId   ///
typedef cudaError_t (*cudaDeviceGetPCIBusId_t)(char* pciBusId, int len, int device);
static cudaDeviceGetPCIBusId_t native_cudaDeviceGetPCIBusId = NULL;

extern "C" cudaError_t cudaDeviceGetPCIBusId (char* pciBusId, int len, int device) {
    printf("\n>>cudaDeviceGetPCIBusId interception\n");

    if (native_cudaDeviceGetPCIBusId == NULL) {
        native_cudaDeviceGetPCIBusId = (cudaDeviceGetPCIBusId_t)dlsym(RTLD_NEXT,"cudaDeviceGetPCIBusId");
    }
    assert(native_cudaDeviceGetPCIBusId != NULL);
    return native_cudaDeviceGetPCIBusId(pciBusId,len,device);
}

///   cudaDeviceGetSharedMemConfig   ///
typedef cudaError_t (*cudaDeviceGetSharedMemConfig_t)( cudaSharedMemConfig ** pConfig );
static cudaDeviceGetSharedMemConfig_t native_cudaDeviceGetSharedMemConfig = NULL;

extern "C" cudaError_t cudaDeviceGetSharedMemConfig (cudaSharedMemConfig ** pConfig ) {
    printf("\n>>cudaDeviceGetSharedMemConfig interception\n");

    if (native_cudaDeviceGetSharedMemConfig == NULL) {
        native_cudaDeviceGetSharedMemConfig = (cudaDeviceGetSharedMemConfig_t)dlsym(RTLD_NEXT,"cudaDeviceGetSharedMemConfig");
    }
    assert(native_cudaDeviceGetSharedMemConfig != NULL);
    return native_cudaDeviceGetSharedMemConfig(pConfig);
}

///   cudaDeviceGetStreamPriorityRange   ///
typedef cudaError_t (*cudaDeviceGetStreamPriorityRange_t)( int* leastPriority, int* greatestPriority);
static cudaDeviceGetStreamPriorityRange_t native_cudaDeviceGetStreamPriorityRange = NULL;

extern "C" cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority) {
    printf("\n>>cudaDeviceGetStreamPriorityRange interception\n");

    if (native_cudaDeviceGetStreamPriorityRange == NULL) {
        native_cudaDeviceGetStreamPriorityRange = (cudaDeviceGetStreamPriorityRange_t)dlsym(RTLD_NEXT,"cudaDeviceGetStreamPriorityRange");
    }
    assert(native_cudaDeviceGetStreamPriorityRange != NULL);
    return native_cudaDeviceGetStreamPriorityRange(leastPriority,greatestPriority);
}

///   cudaMalloc3D   ///
typedef cudaError_t (*cudaDeviceSetCacheConfig_t)(cudaFuncCache cacheConfig);
static cudaDeviceSetCacheConfig_t native_cudaDeviceSetCacheConfig = NULL;

extern "C" cudaError_t cudaDeviceSetCacheConfig (cudaFuncCache cacheConfig) {
    printf("\n>>cudaDeviceSetCacheConfig interception\n");

    if (native_cudaDeviceSetCacheConfig == NULL) {
        native_cudaDeviceSetCacheConfig = (cudaDeviceSetCacheConfig_t)dlsym(RTLD_NEXT,"cudaDeviceSetCacheConfig");
    }
    assert(native_cudaDeviceSetCacheConfig != NULL);
    return native_cudaDeviceSetCacheConfig(cacheConfig);
}

///   cudaDeviceSetLimit   ///
typedef cudaError_t (*cudaDeviceSetLimit_t)(cudaLimit limit, size_t value);
static cudaDeviceSetLimit_t native_cudaDeviceSetLimit = NULL;

extern "C" cudaError_t cudaDeviceSetLimit (cudaLimit limit, size_t value) {
    printf("\n>>cudaDeviceSetLimit interception\n");

    if (native_cudaDeviceSetLimit == NULL) {
        native_cudaDeviceSetLimit = (cudaDeviceSetLimit_t)dlsym(RTLD_NEXT,"cudaDeviceSetLimit");
    }
    assert(native_cudaDeviceSetLimit != NULL);
    return native_cudaDeviceSetLimit(limit,value);
}

///   cudaDeviceSetSharedMemConfig   ///
typedef cudaError_t (*cudaDeviceSetSharedMemConfig_t)(cudaSharedMemConfig config);
static cudaDeviceSetSharedMemConfig_t native_cudaDeviceSetSharedMemConfig = NULL;

extern "C" cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) {
    printf("\n>>cudaDeviceSetSharedMemConfig interception\n");

    if (native_cudaDeviceSetSharedMemConfig == NULL) {
        native_cudaDeviceSetSharedMemConfig = (cudaDeviceSetSharedMemConfig_t)dlsym(RTLD_NEXT,"cudaDeviceSetSharedMemConfig");
    }
    assert(native_cudaDeviceSetSharedMemConfig != NULL);
    return native_cudaDeviceSetSharedMemConfig(config);
}

///   cudaDeviceSynchronize   ///
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
static cudaDeviceSynchronize_t native_cudaDeviceSynchronize = NULL;

extern "C" cudaError_t cudaDeviceSynchronize (void) {
    printf("\n>>cudaDeviceSynchronize interception\n");

    if (native_cudaDeviceSynchronize == NULL) {
        native_cudaDeviceSynchronize = (cudaDeviceSynchronize_t)dlsym(RTLD_NEXT,"cudaDeviceSynchronize");
    }
    assert(native_cudaDeviceSynchronize != NULL);
    return native_cudaDeviceSynchronize();
}

///   cudaGetDevice   ///
typedef cudaError_t (*cudaGetDevice_t)(int *device);
static cudaGetDevice_t native_cudaGetDevice = NULL;

extern "C" cudaError_t cudaGetDevice(int *device){
    printf("\n>>cudaGetDevice \n");
    //call of the real function
    if (native_cudaGetDevice == NULL) {
        native_cudaGetDevice = (cudaGetDevice_t)dlsym(RTLD_NEXT,"cudaGetDevice");
    }
    assert(native_cudaGetDevice != NULL);
    return native_cudaGetDevice(device);
}

///   cudaGetDeviceCount   ///
typedef cudaError_t (*cudaGetDeviceCount_t)(int * count);
static cudaGetDeviceCount_t native_cudaGetDeviceCount = NULL;

extern "C" cudaError_t cudaGetDeviceCount(int * count){
    printf("\n>>cudaGetDeviceCount interception \n");

    if (native_cudaGetDeviceCount == NULL) {
        native_cudaGetDeviceCount = (cudaGetDeviceCount_t)dlsym(RTLD_NEXT,"cudaGetDeviceCount");
    }
    assert(native_cudaGetDeviceCount != NULL);
    return native_cudaGetDeviceCount(count);
}

///   cudaGetDeviceFlags   ///
typedef cudaError_t (*cudaGetDeviceFlags_t)(unsigned int* flags);
static cudaGetDeviceFlags_t native_cudaGetDeviceFlags = NULL;

extern "C" cudaError_t cudaGetDeviceFlags (unsigned int* flags) {
    printf("\n>>cudaGetDeviceFlags interception\n");

    if (native_cudaGetDeviceFlags == NULL) {
        native_cudaGetDeviceFlags = (cudaGetDeviceFlags_t)dlsym(RTLD_NEXT,"cudaGetDeviceFlags");
    }
    assert(native_cudaGetDeviceFlags != NULL);
    return native_cudaGetDeviceFlags(flags);
}

///   cudaGetDeviceProperties   ///
typedef cudaError_t (*cudaGetDeviceProperties_t)(struct cudaDeviceProp * prop, int device);
static cudaGetDeviceProperties_t native_cudaGetDeviceProperties = NULL;

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device){
    printf("\n>>cudaGetDeviceProperties interception \n");

    if (native_cudaGetDeviceProperties == NULL) {
        native_cudaGetDeviceProperties = (cudaGetDeviceProperties_t)dlsym(RTLD_NEXT,"cudaGetDeviceProperties");
    }
    assert(native_cudaGetDeviceProperties != NULL);
    return native_cudaGetDeviceProperties(prop,device);
}

///   cudaIpcCloseMemHandle   ///
typedef cudaError_t (*cudaIpcCloseMemHandle_t)(void* devPtr);
static cudaIpcCloseMemHandle_t native_cudaIpcCloseMemHandle = NULL;

extern "C" cudaError_t cudaIpcCloseMemHandle (void* devPtr) {
    printf("\n>>cudaIpcCloseMemHandle interception\n");

    if (native_cudaIpcCloseMemHandle == NULL) {
        native_cudaIpcCloseMemHandle= (cudaIpcCloseMemHandle_t)dlsym(RTLD_NEXT,"cudaIpcCloseMemHandle");
    }
    assert(native_cudaIpcCloseMemHandle != NULL);
    return native_cudaIpcCloseMemHandle(devPtr);
}

///   cudaIpcGetEventHandle   ///
typedef cudaError_t (*cudaIpcGetEventHandle_t)(cudaIpcEventHandle_t* handle, cudaEvent_t event);
static cudaIpcGetEventHandle_t native_cudaIpcGetEventHandle = NULL;

extern "C" cudaError_t cudaIpcGetEventHandle (cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    printf("\n>>cudaIpcGetEventHandle interception\n");

    if (native_cudaIpcGetEventHandle == NULL) {
        native_cudaIpcGetEventHandle = (cudaIpcGetEventHandle_t)dlsym(RTLD_NEXT,"cudaIpcGetEventHandle");
    }
    assert(native_cudaIpcGetEventHandle != NULL);
    return native_cudaIpcGetEventHandle(handle,event);
}

///   cudaIpcGetMemHandle   ///
typedef cudaError_t (*cudaIpcGetMemHandle_t)(cudaIpcMemHandle_t* handle, void* devPtr);
static cudaIpcGetMemHandle_t native_cudaIpcGetMemHandle= NULL;

extern "C" cudaError_t cudaIpcGetMemHandle (cudaIpcMemHandle_t* handle, void* devPtr) {
    printf("\n>>cudaIpcGetMemHandle interception\n");

    if (native_cudaIpcGetMemHandle == NULL) {
        native_cudaIpcGetMemHandle = (cudaIpcGetMemHandle_t)dlsym(RTLD_NEXT,"cudaIpcGetMemHandle");
    }
    assert(native_cudaIpcGetMemHandle!= NULL);
    return native_cudaIpcGetMemHandle(handle,devPtr);
}

///   cudaIpcOpenEventHandle   ///
typedef cudaError_t (*cudaIpcOpenEventHandle_t)(cudaEvent_t* event, cudaIpcEventHandle_t handle);
static cudaIpcOpenEventHandle_t native_cudaIpcOpenEventHandle = NULL;

extern "C" cudaError_t cudaIpcOpenEventHandle (cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    printf("\n>>cudaIpcOpenEventHandle interception\n");

    if (native_cudaIpcOpenEventHandle== NULL) {
        native_cudaIpcOpenEventHandle = (cudaIpcOpenEventHandle_t)dlsym(RTLD_NEXT,"cudaIpcOpenEventHandle");
    }
    assert(native_cudaIpcOpenEventHandle != NULL);
    return native_cudaIpcOpenEventHandle(event,handle);
}

///   cudaIpcOpenMemHandle   ///
typedef cudaError_t (*cudaIpcOpenMemHandle_t)(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
static cudaIpcOpenMemHandle_t native_cudaIpcOpenMemHandle = NULL;

extern "C" cudaError_t cudaIpcOpenMemHandle (void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    printf("\n>>cudaIpcOpenMemHandle interception\n");

    if (native_cudaIpcOpenMemHandle == NULL) {
        native_cudaIpcOpenMemHandle = (cudaIpcOpenMemHandle_t)dlsym(RTLD_NEXT,"cudaIpcOpenMemHandle");
    }
    assert(native_cudaIpcOpenMemHandle != NULL);
    return native_cudaIpcOpenMemHandle(devPtr,handle,flags);
}

///   cudaSetDevice   ///
typedef cudaError_t (*cudaSetDevice_t)(int device);
static cudaSetDevice_t native_cudaSetDevice = NULL;

extern "C" cudaError_t cudaSetDevice(int device){
    printf("\n>>cudaSetDevice interception \n");

    if (native_cudaSetDevice == NULL) {
        native_cudaSetDevice = (cudaSetDevice_t)dlsym(RTLD_NEXT,"cudaSetDevice");
    }
    assert(native_cudaSetDevice != NULL);
    return native_cudaSetDevice(device);
}

///   cudaSetDeviceFlags   ///
typedef cudaError_t (*cudaSetDeviceFlags_t)(int flags);
static cudaSetDeviceFlags_t native_cudaSetDeviceFlags = NULL;

extern "C" cudaError_t cudaSetDeviceFlags(int flags){
    printf("\n>>cudaSetDeviceFlags interception \n");

    if (native_cudaSetDeviceFlags == NULL) {
        native_cudaSetDeviceFlags = (cudaSetDeviceFlags_t)dlsym(RTLD_NEXT,"cudaSetDeviceFlags");
    }
    assert(native_cudaSetDeviceFlags != NULL);
    return native_cudaSetDeviceFlags(flags);
}

///   cudaSetValidDevices   ///
typedef cudaError_t (*cudaSetValidDevices_t)(int * device_arr, int len);
static cudaSetValidDevices_t native_cudaSetValidDevices = NULL;

extern "C" cudaError_t cudaSetValidDevices(int * device_arr, int len){
    printf("\n>>cudaSetValidDevices interception \n");

    if (native_cudaSetValidDevices == NULL) {
        native_cudaSetValidDevices = (cudaSetValidDevices_t)dlsym(RTLD_NEXT,"cudaSetValidDevices");
    }
    assert(native_cudaSetValidDevices != NULL);
    return native_cudaSetValidDevices(device_arr,len);
}



//**********************************************//
//      CUDA Runtime API Stream Management      //
//**********************************************//
///   cudaStreamAttachMemAsync   ///
typedef cudaError_t (*cudaStreamAttachMemAsync_t)(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags);
static cudaStreamAttachMemAsync_t native_cudaStreamAttachMemAsync = NULL;

extern "C" cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags){
    printf("\n>>cudaStreamAttachMemAsync interception \n");

    if (native_cudaStreamAttachMemAsync == NULL) {
        native_cudaStreamAttachMemAsync = (cudaStreamAttachMemAsync_t)dlsym(RTLD_NEXT,"cudaStreamAttachMemAsync");
    }
    assert(native_cudaStreamAttachMemAsync != NULL);
    return native_cudaStreamAttachMemAsync(stream,devPtr,length,flags);
}


///   cudaStreamCreate   ///
typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t * pStream);
static cudaStreamCreate_t native_cudaStreamCreate = NULL;

extern "C" cudaError_t cudaStreamCreate(cudaStream_t * pStream){
    printf("\n>>cudaStreamCreate interception \n");

    if (native_cudaStreamCreate == NULL) {
        native_cudaStreamCreate = (cudaStreamCreate_t)dlsym(RTLD_NEXT,"cudaStreamCreate");
    }
    assert(native_cudaStreamCreate != NULL);
    return native_cudaStreamCreate(pStream);
}

///   cudaStreamCreateWithFlags   ///
typedef cudaError_t (*cudaStreamCreateWithFlags_t)(cudaStream_t* pStream, unsigned int  flags);
static cudaStreamCreateWithFlags_t native_cudaStreamCreateWithFlags = NULL;

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int  flags){
    printf("\n>>cudaStreamCreateWithFlags interception \n");

    if (native_cudaStreamCreateWithFlags == NULL) {
        native_cudaStreamCreateWithFlags = (cudaStreamCreateWithFlags_t)dlsym(RTLD_NEXT,"cudaStreamCreateWithFlags");
    }
    assert(native_cudaStreamCreateWithFlags != NULL);
    return native_cudaStreamCreateWithFlags(pStream,flags);
}

///   cudaStreamCreateWithPriority   ///
typedef cudaError_t (*cudaStreamCreateWithPriority_t)(cudaStream_t* pStream, unsigned int flags, int priority);
static cudaStreamCreateWithPriority_t native_cudaStreamCreateWithPriority = NULL;

extern "C" cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority){
    printf("\n>>cudaStreamCreateWithPriority interception \n");

    if (native_cudaStreamCreateWithPriority == NULL) {
        native_cudaStreamCreateWithPriority = (cudaStreamCreateWithPriority_t)dlsym(RTLD_NEXT,"cudaStreamCreateWithPriority");
    }
    assert(native_cudaStreamCreateWithPriority != NULL);
    return native_cudaStreamCreateWithPriority(pStream,flags,priority);
}

///   cudaStreamDestroy   ///
typedef cudaError_t (*cudaStreamDestroy_t)(cudaStream_t stream);
static cudaStreamDestroy_t native_cudaStreamDestroy = NULL;

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream){
    printf("\n>>cudaStreamDestroy interception \n");

    if (native_cudaStreamDestroy == NULL) {
        native_cudaStreamDestroy = (cudaStreamDestroy_t)dlsym(RTLD_NEXT,"cudaStreamDestroy");
    }
    assert(native_cudaStreamDestroy != NULL);
    return native_cudaStreamDestroy(stream);
}


///   cudaStreamGetFlags   ///
typedef cudaError_t (*cudaStreamGetFlags_t)(cudaStream_t hStream, unsigned int* flags);
static cudaStreamGetFlags_t native_cudaStreamGetFlags= NULL;

extern "C" cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags){
    printf("\n>>cudaStreamGetFlags interception \n");

    if (native_cudaStreamGetFlags == NULL) {
        native_cudaStreamGetFlags = (cudaStreamGetFlags_t)dlsym(RTLD_NEXT,"cudaStreamGetFlags");
    }
    assert(native_cudaStreamGetFlags != NULL);
    return native_cudaStreamGetFlags(hStream,flags);
}

///   cudaStreamGetPriority   ///
typedef cudaError_t (*cudaStreamGetPriority_t)(cudaStream_t hStream, int* priority);
static cudaStreamGetPriority_t native_cudaStreamGetPriority = NULL;

extern "C" cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority){
    printf("\n>>cudaStreamGetPriority interception \n");

    if (native_cudaStreamGetPriority == NULL) {
        native_cudaStreamGetPriority = (cudaStreamGetPriority_t)dlsym(RTLD_NEXT,"cudaStreamGetPriority");
    }
    assert(native_cudaStreamGetPriority != NULL);
    return native_cudaStreamGetPriority(hStream,priority);
}

///   cudaStreamQuery   ///
typedef cudaError_t (*cudaStreamQuery_t)(cudaStream_t stream);
static cudaStreamQuery_t native_cudaStreamQuery = NULL;

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream){
    printf("\n>>cudaStreamQuery interception \n");

    if (native_cudaStreamQuery == NULL) {
        native_cudaStreamQuery = (cudaStreamQuery_t)dlsym(RTLD_NEXT,"cudaStreamQuery");
    }
    assert(native_cudaStreamQuery != NULL);
    return native_cudaStreamQuery(stream);
}

///   cudaStreamSynchronize   ///
typedef cudaError_t (*cudaStreamSynchronize_t)(cudaStream_t stream);
static cudaStreamSynchronize_t native_cudaStreamSynchronize = NULL;

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream){
    printf("\n>>cudaStreamSynchronize interception \n");

    if (native_cudaStreamSynchronize== NULL) {
        native_cudaStreamSynchronize = (cudaStreamSynchronize_t)dlsym(RTLD_NEXT,"cudaStreamSynchronize");
    }
    assert(native_cudaStreamSynchronize != NULL);
    return native_cudaStreamSynchronize(stream);
}

///   cudaStreamWaitEvent   ///
typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
static cudaStreamWaitEvent_t native_cudaStreamWaitEvent = NULL;

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags){
    printf("\n>>cudaStreamWaitEvent interception \n");

    if (native_cudaStreamWaitEvent == NULL) {
        native_cudaStreamWaitEvent = (cudaStreamWaitEvent_t)dlsym(RTLD_NEXT,"cudaStreamWaitEvent");
    }
    assert(native_cudaStreamWaitEvent != NULL);
    return native_cudaStreamWaitEvent(stream,event,flags);
}



//*********************************************//
//      CUDA Runtime API Event Management      //
//*********************************************//
///   cudaDriverGetVersion   ///
typedef cudaError_t (*cudaEventCreate_t)(cudaEvent_t * event);
static cudaEventCreate_t native_cudaEventCreate = NULL;

extern "C" cudaError_t cudaEventCreate (cudaEvent_t * event) {
    printf("\n>>cudaEventCreate interception\n");

    if (native_cudaEventCreate == NULL) {
        native_cudaEventCreate = (cudaEventCreate_t)dlsym(RTLD_NEXT,"cudaEventCreate");
    }
    assert(native_cudaEventCreate != NULL);
    return native_cudaEventCreate(event);
}

///   cudaEventCreateWithFlags   ///
typedef cudaError_t (*cudaEventCreateWithFlags_t)(cudaEvent_t * event, int flags);
static cudaEventCreateWithFlags_t native_cudaEventCreateWithFlags = NULL;

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, int flags) {
    printf("\n>>cudaEventCreateWithFlags interception\n");

    if (native_cudaEventCreateWithFlags == NULL) {
        native_cudaEventCreateWithFlags = (cudaEventCreateWithFlags_t)dlsym(RTLD_NEXT,"cudaEventCreateWithFlags");
    }
    assert(native_cudaEventCreateWithFlags != NULL);
    return native_cudaEventCreateWithFlags(event,flags);
}

///   cudaEventDestroy   ///
typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t event);
static cudaEventDestroy_t native_cudaEventDestroy = NULL;

extern "C" cudaError_t cudaEventDestroy	(cudaEvent_t event) {
    printf("\n>>cudaEventDestroy interception\n");

    if (native_cudaEventDestroy == NULL) {
        native_cudaEventDestroy = (cudaEventDestroy_t)dlsym(RTLD_NEXT,"cudaEventDestroy");
    }

    assert(native_cudaEventDestroy != NULL);
    return native_cudaEventDestroy(event);
}

///   cudaEventElapsedTime   ///
typedef cudaError_t (*cudaEventElapsedTime_t)(float * ms, cudaEvent_t start, cudaEvent_t end);
static cudaEventElapsedTime_t native_cudaEventElapsedTime = NULL;

extern "C" cudaError_t cudaEventElapsedTime	(float * ms, cudaEvent_t start,cudaEvent_t end) {
    printf("\n>>cudaEventElapsedTime interception\n");

    if (native_cudaEventElapsedTime == NULL) {
        native_cudaEventElapsedTime = (cudaEventElapsedTime_t)dlsym(RTLD_NEXT,"cudaEventElapsedTime");
    }
    assert(native_cudaEventElapsedTime != NULL);
    return native_cudaEventElapsedTime(ms,start,end);
}

///   cudaEventQuery   ///
typedef cudaError_t (*cudaEventQuery_t)(cudaEvent_t event);
static cudaEventQuery_t native_cudaEventQuery = NULL;

extern "C" cudaError_t cudaEventQuery (cudaEvent_t event) {
    printf("\n>>cudaEventQuery interception\n");

    if (native_cudaEventQuery == NULL) {
        native_cudaEventQuery = (cudaEventQuery_t)dlsym(RTLD_NEXT,"cudaEventQuery");
    }
    assert(native_cudaEventQuery != NULL);
    return native_cudaEventQuery(event);
}

///   cudaEventRecord   ///
typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t event, cudaStream_t stream);
static cudaEventRecord_t native_cudaEventRecord = NULL;

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    printf("\n>>cudaEventRecord interception\n");

    if (native_cudaEventRecord == NULL) {
        native_cudaEventRecord = (cudaEventRecord_t)dlsym(RTLD_NEXT,"cudaEventRecord");
    }
    assert(native_cudaEventRecord != NULL);
    return native_cudaEventRecord(event,stream);
}

///   cudaEventSynchronize   ///
typedef cudaError_t (*cudaEventSynchronize_t)(cudaEvent_t event);
static cudaEventSynchronize_t native_cudaEventSynchronize = NULL;

extern "C" cudaError_t cudaEventSynchronize	(cudaEvent_t event) {
    printf("\n>>cudaEventSynchronize interception\n");

    if (native_cudaEventSynchronize == NULL) {
        native_cudaEventSynchronize = (cudaEventSynchronize_t)dlsym(RTLD_NEXT,"cudaEventSynchronize");
    }
    assert(native_cudaEventSynchronize != NULL);
    return native_cudaEventSynchronize(event);
}


//**********************************************//
//      CUDA Runtime API Execution Control      //
//**********************************************//
//  cudaConfigureCall  ///
typedef cudaError_t (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
static cudaConfigureCall_t native_CudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem=0, cudaStream_t stream=0) {
    assert(kernelInfo().counter == 0);
    kernelInfo().gridDim = gridDim;
    kernelInfo().blockDim = blockDim;
    //kernelInfo().counter++;   //increase a counter to indicate an expected cudaLaunch to be completed
    printf("\n>>cudaConfigureCall interception\n");
    //call of the real function
    if (native_CudaConfigureCall == NULL)
        native_CudaConfigureCall = (cudaConfigureCall_t)dlsym(RTLD_NEXT,"cudaConfigureCall");

    assert(native_CudaConfigureCall != NULL);
    return native_CudaConfigureCall(gridDim,blockDim,sharedMem,stream);
}


///   cudaFuncGetAttributes   ///
typedef cudaError_t (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes * attr, const char * func);
static cudaFuncGetAttributes_t native_cudaFuncGetAttributes = NULL;

extern "C" cudaError_t cudaFuncGetAttributes (struct cudaFuncAttributes * attr, const char * func) {
    printf("\n>>cudaFuncGetAttributes interception\n");

    if (native_cudaFuncGetAttributes == NULL) {
        native_cudaFuncGetAttributes = (cudaFuncGetAttributes_t)dlsym(RTLD_NEXT,"cudaFuncGetAttributes");
    }
    assert(native_cudaFuncGetAttributes != NULL);
    return native_cudaFuncGetAttributes(attr,func);
}

///   cudaFuncSetAttribute   ///
typedef cudaError_t (*cudaFuncSetAttribute_t)(const void* func, cudaFuncAttribute attr, int  value);
static cudaFuncSetAttribute_t native_cudaFuncSetAttribute = NULL;

extern "C" cudaError_t cudaFuncSetAttribute (const void* func, cudaFuncAttribute attr, int  value) {
    printf("\n>>cudaFuncSetAttribute interception\n");

    if (native_cudaFuncSetAttribute == NULL) {
        native_cudaFuncSetAttribute = (cudaFuncSetAttribute_t)dlsym(RTLD_NEXT,"cudaFuncSetAttribute");
    }
    assert(native_cudaFuncSetAttribute != NULL);
    return native_cudaFuncSetAttribute(func,attr,value);
}

///  cudaLaunch ///
typedef cudaError_t (*cudaLaunch_t)(const char* entry);
static cudaLaunch_t native_cudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch( const char* entry){
     //print_kernel_invocation(entry);
     //kernelInfo().counter--;
     printf("\n>>cudaLaunch interception\n");
    //call of the real function
    if (native_cudaLaunch == NULL) {
        native_cudaLaunch = (cudaLaunch_t)dlsym(RTLD_NEXT,"cudaLaunch");
    }
    assert(native_cudaLaunch != NULL);
    return native_cudaLaunch(entry);
}


///   cudaFuncSetCacheConfig   ///
typedef cudaError_t (*cudaFuncSetCacheConfig_t)(const void* func, cudaFuncCache cacheConfig);
static cudaFuncSetCacheConfig_t native_cudaFuncSetCacheConfig = NULL;

extern "C" cudaError_t cudaFuncSetCacheConfig (const void* func, cudaFuncCache cacheConfig) {
    printf("\n>>cudaFuncSetCacheConfig interception\n");

    if (native_cudaFuncSetCacheConfig == NULL) {
        native_cudaFuncSetCacheConfig = (cudaFuncSetCacheConfig_t)dlsym(RTLD_NEXT,"cudaFuncSetCacheConfig");
    }
    assert(native_cudaFuncSetCacheConfig != NULL);
    return native_cudaFuncSetCacheConfig(func,cacheConfig);
}

///   cudaFuncSetSharedMemConfig   ///
typedef cudaError_t (*cudaFuncSetSharedMemConfig_t)(const void* func, cudaSharedMemConfig config);
static cudaFuncSetSharedMemConfig_t native_cudaFuncSetSharedMemConfig = NULL;

extern "C" cudaError_t cudaFuncSetSharedMemConfig (const void* func, cudaSharedMemConfig config) {
    printf("\n>>cudaFuncSetSharedMemConfig interception\n");

    if (native_cudaFuncSetSharedMemConfig == NULL) {
        native_cudaFuncSetSharedMemConfig = (cudaFuncSetSharedMemConfig_t)dlsym(RTLD_NEXT,"cudaFuncSetSharedMemConfig");
    }
    assert(native_cudaFuncSetSharedMemConfig != NULL);
    return native_cudaFuncSetSharedMemConfig(func,config);
}

///   cudaGetParameterBuffer   ///
typedef cudaError_t (*cudaGetParameterBuffer_t)(size_t alignment, size_t size);
static cudaGetParameterBuffer_t native_cudaGetParameterBuffer = NULL;

extern "C" cudaError_t cudaGetParameterBuffer (size_t alignment, size_t size) {
    printf("\n>>cudaGetParameterBuffer interception\n");

    if (native_cudaGetParameterBuffer == NULL) {
        native_cudaGetParameterBuffer = (cudaGetParameterBuffer_t)dlsym(RTLD_NEXT,"cudaGetParameterBuffer");
    }
    assert(native_cudaGetParameterBuffer != NULL);
    return native_cudaGetParameterBuffer(alignment,size);
}

///   cudaGetParameterBufferV2   ///
typedef cudaError_t (*cudaGetParameterBufferV2_t)(void* func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
static cudaGetParameterBufferV2_t native_cudaGetParameterBufferV2 = NULL;

extern "C" cudaError_t cudaGetParameterBufferV2	(void* func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize) {
    printf("\n>>cudaGetParameterBufferV2 interception\n");

    if (native_cudaGetParameterBufferV2 == NULL) {
        native_cudaGetParameterBufferV2 = (cudaGetParameterBufferV2_t)dlsym(RTLD_NEXT,"cudaGetParameterBufferV2");
    }
    assert(native_cudaGetParameterBufferV2 != NULL);
    return native_cudaGetParameterBufferV2(func,gridDimension,blockDimension,sharedMemSize);
}

///   cudaLaunchCooperativeKernel   ///
typedef cudaError_t (*cudaLaunchCooperativeKernel_t)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
static cudaLaunchCooperativeKernel_t native_cudaLaunchCooperativeKernel = NULL;

extern "C" cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    printf("\n>>cudaLaunchCooperativeKernel interception\n");

    if (native_cudaLaunchCooperativeKernel == NULL) {
        native_cudaLaunchCooperativeKernel = (cudaLaunchCooperativeKernel_t)dlsym(RTLD_NEXT,"cudaLaunchCooperativeKernel");
    }
    assert(native_cudaLaunchCooperativeKernel != NULL);
    return native_cudaLaunchCooperativeKernel(func,gridDim,blockDim,args,sharedMem,stream);
}

///   cudaLaunchCooperativeKernelMultiDevice   ///
typedef cudaError_t (*cudaLaunchCooperativeKernelMultiDevice_t)(cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags);
static cudaLaunchCooperativeKernelMultiDevice_t native_cudaLaunchCooperativeKernelMultiDevice = NULL;

extern "C" cudaError_t cudaLaunchCooperativeKernelMultiDevice (cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags) {
    printf("\n>>cudaLaunchCooperativeKernelMultiDevice interception\n");

    if (native_cudaLaunchCooperativeKernelMultiDevice == NULL) {
        native_cudaLaunchCooperativeKernelMultiDevice = (cudaLaunchCooperativeKernelMultiDevice_t)dlsym(RTLD_NEXT,"cudaLaunchCooperativeKernelMultiDevice");
    }
    assert(native_cudaLaunchCooperativeKernelMultiDevice != NULL);
    return native_cudaLaunchCooperativeKernelMultiDevice(launchParamsList,numDevices,flags);
}



///   cudaLaunchKernel   ///
typedef cudaError_t (*cudaLaunchKernel_t)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
static cudaLaunchKernel_t native_cudaLaunchKernel = NULL;

extern "C" cudaError_t cudaLaunchKernel	(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    printf("\n>>cudaLaunchKernel interception\n");

    if (native_cudaLaunchKernel == NULL) {
        native_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(RTLD_NEXT,"cudaLaunchKernel");
    }
    assert(native_cudaLaunchKernel != NULL);
    return native_cudaLaunchKernel(func,gridDim,blockDim,args,sharedMem,stream);
}

///   cudaSetDoubleForDevice   ///
typedef cudaError_t (*cudaSetDoubleForDevice_t)(double *d);
static cudaSetDoubleForDevice_t native_cudaSetDoubleForDevice = NULL;

extern "C" cudaError_t cudaSetDoubleForDevice (double *d) {
    printf("\n>>cudaSetDoubleForDevice interception\n");

    if (native_cudaSetDoubleForDevice == NULL) {
        native_cudaSetDoubleForDevice = (cudaSetDoubleForDevice_t)dlsym(RTLD_NEXT,"cudaSetDoubleForDevice");
    }
    assert(native_cudaSetDoubleForDevice != NULL);
    return native_cudaSetDoubleForDevice(d);
}

///   cudaSetDoubleForHost   ///
typedef cudaError_t (*cudaSetDoubleForHost_t)(double *d);
static cudaSetDoubleForHost_t native_cudaSetDoubleForHost = NULL;

extern "C" cudaError_t cudaSetDoubleForHost	(double *d) {
    printf("\n>>cudaSetDoubleForHost interception\n");

    if (native_cudaSetDoubleForHost == NULL) {
        native_cudaSetDoubleForHost = (cudaSetDoubleForHost_t)dlsym(RTLD_NEXT,"cudaSetDoubleForHost");
    }
    assert(native_cudaSetDoubleForHost != NULL);
    return native_cudaSetDoubleForHost(d);
}

/*  cudaSetupArgument   ///
typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
static cudaSetupArgument_t native_CudaSetupArgument = NULL;

extern "C" cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    kernelInfo().arguments.push_back(const_cast<void *>(arg));

    //call of the real function
    if (native_CudaSetupArgument == NULL) {
        native_CudaSetupArgument = (cudaSetupArgument_t)dlsym(RTLD_NEXT,"cudaSetupArgument");
    }
    assert(native_CudaSetupArgument != NULL);
    return native_CudaSetupArgument(arg, size, offset);
}
*/


//**********************************************//
//      CUDA Runtime API Memory Management      //
//**********************************************//
///   cudaFree   ///
typedef cudaError_t (*cudaFree_t)(void * devPtr);
static cudaFree_t native_cudaFree = NULL;

extern "C" cudaError_t cudaFree	(void * devPtr) {
    printf("\n>>cudaFree interception\n");

    if (native_cudaFree == NULL) {
        native_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT,"cudaFree");
    }
    assert(native_cudaFree != NULL);
    return native_cudaFree(devPtr);
}


///   cudaFreeArray   ///
typedef cudaError_t (*cudaFreeArray_t)(struct cudaArray * array);
static cudaFreeArray_t native_cudaFreeArray = NULL;

extern "C" cudaError_t cudaFreeArray (struct cudaArray * array) {
    printf("\n>>cudaFreeArray interception\n");

    if (native_cudaFreeArray == NULL) {
        native_cudaFreeArray = (cudaFreeArray_t)dlsym(RTLD_NEXT,"cudaFreeArray");
    }
    assert(native_cudaFreeArray != NULL);
    return native_cudaFreeArray(array);
}


///   cudaFreeHost   ///
typedef cudaError_t (*cudaFreeHost_t)(void * ptr);
static cudaFreeHost_t native_cudaFreeHost = NULL;

extern "C" cudaError_t cudaFreeHost(void * ptr) {
    printf("\n>>cudaFreeHost interception\n");

    if (native_cudaFreeHost == NULL) {
        native_cudaFreeHost = (cudaFreeHost_t)dlsym(RTLD_NEXT,"cudaFreeHost");
    }
    assert(native_cudaFreeHost != NULL);
    return native_cudaFreeHost(ptr);
}


///   cudaGetSymbolAddress   ///
typedef cudaError_t (*cudaGetSymbolAddress_t)(void ** devPtr, const char * symbol);
static cudaGetSymbolAddress_t native_cudaGetSymbolAddress = NULL;

extern "C" cudaError_t cudaGetSymbolAddress	(void ** devPtr, const char * symbol) {
    printf("\n>>cudaGetSymbolAddress interception\n");

    if (native_cudaGetSymbolAddress == NULL) {
        native_cudaGetSymbolAddress = (cudaGetSymbolAddress_t)dlsym(RTLD_NEXT,"cudaGetSymbolAddress");
    }
    assert(native_cudaGetSymbolAddress != NULL);
    return native_cudaGetSymbolAddress(devPtr,symbol);
}


///   cudaGetSymbolSize   ///
typedef cudaError_t (*cudaGetSymbolSize_t)(size_t * size, const char * symbol);
static cudaGetSymbolSize_t native_cudaGetSymbolSize = NULL;

extern "C" cudaError_t cudaGetSymbolSize(size_t * size, const char * symbol) {
    printf("\n>>cudaGetSymbolSize interception\n");

    if (native_cudaGetSymbolSize == NULL) {
        native_cudaGetSymbolSize = (cudaGetSymbolSize_t)dlsym(RTLD_NEXT,"cudaGetSymbolSize");
    }
    assert(native_cudaGetSymbolSize != NULL);
    return native_cudaGetSymbolSize(size,symbol);
}


///   cudaHostAlloc   ///
typedef cudaError_t (*cudaHostAlloc_t)(void ** ptr, size_t size, unsigned int flags);
static cudaHostAlloc_t native_cudaHostAlloc = NULL;

extern "C" cudaError_t cudaHostAlloc (void ** ptr, size_t size, unsigned int flags) {
    printf("\n>>cudaHostAlloc interception\n");

    if (native_cudaHostAlloc == NULL) {
        native_cudaHostAlloc = (cudaHostAlloc_t)dlsym(RTLD_NEXT,"cudaHostAlloc");
    }
    assert(native_cudaHostAlloc != NULL);
    return native_cudaHostAlloc(ptr,size,flags);
}


///   cudaHostGetDevicePointer   ///
typedef cudaError_t (*cudaHostGetDevicePointer_t)(void ** pDevice, void * pHost, unsigned int flags);
static cudaHostGetDevicePointer_t native_cudaHostGetDevicePointer = NULL;

extern "C" cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags) {
    printf("\n>>cudaHostGetDevicePointer interception\n");

    if (native_cudaHostGetDevicePointer == NULL) {
        native_cudaHostGetDevicePointer = (cudaHostGetDevicePointer_t)dlsym(RTLD_NEXT,"cudaHostGetDevicePointer");
    }
    assert(native_cudaHostGetDevicePointer != NULL);
    return native_cudaHostGetDevicePointer(pDevice,pHost,flags);
}


///   cudaHostGetFlags   ///
typedef cudaError_t (*cudaHostGetFlags_t)(unsigned int * pFlags, void * pHost);
static cudaHostGetFlags_t native_cudaHostGetFlags = NULL;

extern "C" cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost) {
    printf("\n>>cudaHostGetFlags interception\n");

    if (native_cudaHostGetFlags == NULL) {
        native_cudaHostGetFlags = (cudaHostGetFlags_t)dlsym(RTLD_NEXT,"cudaHostGetFlags");
    }
    assert(native_cudaHostGetFlags != NULL);
    return native_cudaHostGetFlags(pFlags,pHost);
}


///   cudaMalloc   ///
typedef cudaError_t (*cudaMalloc_t)(void ** devPtr, size_t size);
static cudaMalloc_t native_cudaMalloc = NULL;

extern "C" cudaError_t cudaMalloc(void ** devPtr, size_t size) {
    printf("\n>>cudaMalloc interception\n");

    if (native_cudaMalloc == NULL) {
        native_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_NEXT,"cudaMalloc");
    }
    assert(native_cudaMalloc != NULL);
    return native_cudaMalloc(devPtr,size);
}


///   cudaMalloc3D   ///
typedef cudaError_t (*cudaMalloc3D_t)(struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent);
static cudaMalloc3D_t native_cudaMalloc3D = NULL;

extern "C" cudaError_t cudaMalloc3D (struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent) {
    printf("\n>>cudaMalloc3D interception\n");

    if (native_cudaMalloc3D == NULL) {
        native_cudaMalloc3D = (cudaMalloc3D_t)dlsym(RTLD_NEXT,"cudaMalloc3D");
    }
    assert(native_cudaMalloc3D != NULL);
    return native_cudaMalloc3D(pitchedDevPtr,extent);
}


///   cudaMalloc3DArray   ///
typedef cudaError_t (*cudaMalloc3DArray_t)(struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent);
static cudaMalloc3DArray_t native_cudaMalloc3DArray = NULL;

extern "C" cudaError_t cudaMalloc3DArray (struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent) {
    printf("\n>>cudaMalloc3DArray interception\n");

    if (native_cudaMalloc3DArray == NULL) {
        native_cudaMalloc3DArray = (cudaMalloc3DArray_t)dlsym(RTLD_NEXT,"cudaMalloc3DArray");
    }
    assert(native_cudaMalloc3DArray != NULL);
    return native_cudaMalloc3DArray(arrayPtr,desc,extent);
}


///   cudaMallocArray   ///
typedef cudaError_t (*cudaMallocArray_t)(struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height);
static cudaMallocArray_t native_cudaMallocArray = NULL;

extern "C" cudaError_t cudaMallocArray (struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height) {
    printf("\n>>cudaMallocArray interception\n");

    if (native_cudaMallocArray == NULL) {
        native_cudaMallocArray = (cudaMallocArray_t)dlsym(RTLD_NEXT,"cudaMallocArray");
    }
    assert(native_cudaMallocArray != NULL);
    return native_cudaMallocArray(arrayPtr,desc,width,height);
}


///   cudaMallocHost   ///
typedef cudaError_t (*cudaMallocHost_t)(void ** ptr,size_t size);
static cudaMallocHost_t native_cudaMallocHost = NULL;

extern "C" cudaError_t cudaMallocHost (void ** ptr,size_t size) {
    printf("\n>>cudaMallocHost interception\n");

    if (native_cudaMallocHost == NULL) {
        native_cudaMallocHost = (cudaMallocHost_t)dlsym(RTLD_NEXT,"cudaMallocHost");
    }
    assert(native_cudaMallocHost != NULL);
    return native_cudaMallocHost(ptr,size);
}


///   cudaMallocPitch   ///
typedef cudaError_t (*cudaMallocPitch_t)(void ** devPtr, size_t * pitch, size_t width, size_t height);
static cudaMallocPitch_t native_cudaMallocPitch = NULL;

extern "C" cudaError_t cudaMallocPitch (void ** devPtr, size_t * pitch, size_t width, size_t height) {
    printf("\n>>cudaMallocPitch interception\n");

    if (native_cudaMallocPitch == NULL) {
        native_cudaMallocPitch = (cudaMallocPitch_t)dlsym(RTLD_NEXT,"cudaMallocPitch");
    }
    assert(native_cudaMallocPitch != NULL);
    return native_cudaMallocPitch(devPtr,pitch,width,height);
}


///   cudaMemcpy   ///
typedef cudaError_t (*cudaMemcpy_t)(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);
static cudaMemcpy_t native_cudaMemcpy = NULL;

extern "C" cudaError_t cudaMemcpy (void * dst, const void * src, size_t count, enum cudaMemcpyKind kind) {
    printf("\n>>cudaMemcpy interception\n");

    if (native_cudaMemcpy == NULL) {
        native_cudaMemcpy = (cudaMemcpy_t)dlsym(RTLD_NEXT,"cudaMemcpy");
    }
    assert(native_cudaMemcpy != NULL);
    return native_cudaMemcpy(dst,src,count,kind);
}


///   cudaMemcpy2D   ///
typedef cudaError_t (*cudaMemcpy2D_t)(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
static cudaMemcpy2D_t native_cudaMemcpy2D= NULL;

extern "C" cudaError_t cudaMemcpy2D (void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    printf("\n>>cudaMemcpy2D interception\n");

    if (native_cudaMemcpy2D == NULL) {
        native_cudaMemcpy2D = (cudaMemcpy2D_t)dlsym(RTLD_NEXT,"cudaMemcpy2D");
    }
    assert(native_cudaMemcpy2D != NULL);
    return native_cudaMemcpy2D(dst,dpitch,src,spitch,width,height,kind);
}


///   cudaMemcpy2DArrayToArray   ///
typedef cudaError_t (*cudaMemcpy2DArrayToArray_t)(struct cudaArray * dst,
    size_t 	wOffsetDst,
    size_t 	hOffsetDst,
    const struct cudaArray * src,
    size_t 	wOffsetSrc,
    size_t 	hOffsetSrc,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind);

static cudaMemcpy2DArrayToArray_t native_cudaMemcpy2DArrayToArray = NULL;

extern "C" cudaError_t cudaMemcpy2DArrayToArray (struct cudaArray * dst,
    size_t 	wOffsetDst,
    size_t 	hOffsetDst,
    const struct cudaArray * src,
    size_t 	wOffsetSrc,
    size_t 	hOffsetSrc,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind) {
    printf("\n>>cudaMalloc3D interception\n");

    if (native_cudaMemcpy2DArrayToArray == NULL) {
        native_cudaMemcpy2DArrayToArray = (cudaMemcpy2DArrayToArray_t)dlsym(RTLD_NEXT,"cudaMemcpy2DArrayToArray");
    }
    assert(native_cudaMemcpy2DArrayToArray != NULL);
    return native_cudaMemcpy2DArrayToArray(dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,width,height,kind);
}


///   cudaMemcpy2DAsync   ///
typedef cudaError_t (*cudaMemcpy2DAsync_t)(void * dst,
    size_t 	dpitch,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpy2DAsync_t native_cudaMemcpy2DAsync = NULL;

extern "C" cudaError_t cudaMemcpy2DAsync (void * dst,
    size_t 	dpitch,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    printf("\n>>cudaMemcpy2DAsync interception\n");

    if (native_cudaMemcpy2DAsync == NULL) {
        native_cudaMemcpy2DAsync = (cudaMemcpy2DAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy2DAsync");
    }
    assert(native_cudaMemcpy2DAsync != NULL);
    return native_cudaMemcpy2DAsync(dst,dpitch,src,spitch,width,height,kind,stream);
}


///   cudaMemcpy2DFromArray   ///
typedef cudaError_t (*cudaMemcpy2DFromArray_t)(void * dst,
    size_t 	dpitch,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind);

static cudaMemcpy2DFromArray_t native_cudaMemcpy2DFromArray = NULL;

extern "C" cudaError_t cudaMemcpy2DFromArray (void * dst,
    size_t 	dpitch,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind){

    printf("\n>>cudaMemcpy2DFromArray interception\n");

    if (native_cudaMemcpy2DFromArray == NULL) {
        native_cudaMemcpy2DFromArray = (cudaMemcpy2DFromArray_t)dlsym(RTLD_NEXT,"cudaMemcpy2DFromArray");
    }
    assert(native_cudaMemcpy2DFromArray != NULL);
    return native_cudaMemcpy2DFromArray(dst,dpitch,src,wOffset,hOffset,width,height,kind);
}



///   cudaMemcpy2DFromArrayAsync   ///
typedef cudaError_t (*cudaMemcpy2DFromArrayAsync_t)(void * dst,
    size_t 	dpitch,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpy2DFromArrayAsync_t native_cudaMemcpy2DFromArrayAsync = NULL;

extern "C" cudaError_t cudaMemcpy2DFromArrayAsync (void * dst,
    size_t 	dpitch,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream){

    printf("\n>>cudaMemcpy2DFromArrayAsync interception\n");

    if (native_cudaMemcpy2DFromArrayAsync == NULL) {
        native_cudaMemcpy2DFromArrayAsync = (cudaMemcpy2DFromArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy2DFromArrayAsync");
    }
    assert(native_cudaMemcpy2DFromArrayAsync != NULL);
    return native_cudaMemcpy2DFromArrayAsync(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream);
}










///   cudaMemcpy2DToArray   ///
typedef cudaError_t (*cudaMemcpy2DToArray_t)(struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind);

static cudaMemcpy2DToArray_t native_cudaMemcpy2DToArray= NULL;

extern "C" cudaError_t cudaMemcpy2DToArray (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind) {

    printf("\n>>cudaMemcpy2DToArray interception\n");

    if (native_cudaMemcpy2DToArray == NULL) {
        native_cudaMemcpy2DToArray = (cudaMemcpy2DToArray_t)dlsym(RTLD_NEXT,"cudaMemcpy2DToArray");
    }
    assert(native_cudaMemcpy2DToArray != NULL);
    return native_cudaMemcpy2DToArray(dst,wOffset,hOffset,src,spitch,width,height,kind);
}


///   cudaMemcpy2DToArrayAsync   ///
typedef cudaError_t (*cudaMemcpy2DToArrayAsync_t)(struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpy2DToArrayAsync_t native_cudaMemcpy2DToArrayAsync = NULL;

extern "C" cudaError_t cudaMemcpy2DToArrayAsync (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {

    printf("\n>>cudaMemcpy2DToArrayAsync interception\n");

    if (native_cudaMemcpy2DToArrayAsync == NULL) {
        native_cudaMemcpy2DToArrayAsync = (cudaMemcpy2DToArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy2DToArrayAsync");
    }
    assert(native_cudaMemcpy2DToArrayAsync != NULL);
    return native_cudaMemcpy2DToArrayAsync(dst,wOffset,hOffset,src,spitch,width,height,kind,stream);
}


///   cudaMemcpy3D   ///
typedef cudaError_t (*cudaMemcpy3D_t)(const struct cudaMemcpy3DParms * p);
static cudaMemcpy3D_t native_cudaMemcpy3D = NULL;

extern "C" cudaError_t cudaMemcpy3D (const struct cudaMemcpy3DParms * p) {
    printf("\n>>cudaMemcpy3D interception\n");

    if (native_cudaMemcpy3D== NULL) {
        native_cudaMemcpy3D = (cudaMemcpy3D_t)dlsym(RTLD_NEXT,"cudaMemcpy3D");
    }
    assert(native_cudaMemcpy3D != NULL);
    return native_cudaMemcpy3D(p);
}


///   cudaMemcpy3DAsync   ///
typedef cudaError_t (*cudaMemcpy3DAsync_t)(const struct cudaMemcpy3DParms * p, cudaStream_t stream);
static cudaMemcpy3DAsync_t native_cudaMemcpy3DAsync = NULL;

extern "C" cudaError_t cudaMemcpy3DAsync (const struct cudaMemcpy3DParms * p, cudaStream_t stream) {
    printf("\n>>cudaMemcpy3DAsync interception\n");

    if (native_cudaMemcpy3DAsync == NULL) {
        native_cudaMemcpy3DAsync = (cudaMemcpy3DAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy3DAsync");
    }
    assert(native_cudaMemcpy3DAsync != NULL);
    return native_cudaMemcpy3DAsync(p,stream);
}


///   cudaMemcpyArrayToArray   ///
typedef cudaError_t (*cudaMemcpyArrayToArray_t)(struct cudaArray * dst,
    size_t 	wOffsetDst,
    size_t 	hOffsetDst,
    const struct cudaArray * src,
    size_t 	wOffsetSrc,
    size_t 	hOffsetSrc,
    size_t 	count,
    enum cudaMemcpyKind kind);

static cudaMemcpyArrayToArray_t native_cudaMemcpyArrayToArray = NULL;

extern "C" cudaError_t cudaMemcpyArrayToArray(struct cudaArray * dst,
    size_t 	wOffsetDst,
    size_t 	hOffsetDst,
    const struct cudaArray * src,
    size_t 	wOffsetSrc,
    size_t 	hOffsetSrc,
    size_t 	count,
    enum cudaMemcpyKind kind){

    printf("\n>>cudaMemcpyArrayToArray interception\n");

    if (native_cudaMemcpyArrayToArray == NULL) {
        native_cudaMemcpyArrayToArray = (cudaMemcpyArrayToArray_t)dlsym(RTLD_NEXT,"cudaMemcpyArrayToArray");
    }
    assert(native_cudaMemcpyArrayToArray != NULL);
    return native_cudaMemcpyArrayToArray(dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,count,kind);
}


///   cudaMemcpyAsync   ///
typedef cudaError_t (*cudaMemcpyAsync_t)(void * dst,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpyAsync_t native_cudaMemcpyAsync = NULL;

extern "C" cudaError_t cudaMemcpyAsync (void * dst,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {

    printf("\n>>cudaMemcpyAsync interception\n");

    if (native_cudaMemcpyAsync == NULL) {
        native_cudaMemcpyAsync = (cudaMemcpyAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyAsync");
    }
    assert(native_cudaMemcpyAsync != NULL);
    return native_cudaMemcpyAsync(dst,src,count,kind,stream);
}


///   cudaMemcpyFromArray   ///
typedef cudaError_t (*cudaMemcpyFromArray_t)(void * dst,
    const struct cudaArray * src,
    size_t wOffset,
    size_t hOffset,
    size_t count,
    enum cudaMemcpyKind kind);

static cudaMemcpyFromArray_t native_cudaMemcpyFromArray = NULL;

extern "C" cudaError_t cudaMemcpyFromArray (void * dst,
    const struct cudaArray * src,
    size_t wOffset,
    size_t hOffset,
    size_t count,
    enum cudaMemcpyKind kind){

    printf("\n>>cudaMemcpyFromArray interception\n");

    if (native_cudaMemcpyFromArray == NULL) {
        native_cudaMemcpyFromArray = (cudaMemcpyFromArray_t)dlsym(RTLD_NEXT,"cudaMemcpyFromArray");
    }
    assert(native_cudaMemcpyFromArray != NULL);
    return native_cudaMemcpyFromArray(dst,src,wOffset,hOffset,count,kind);
}


///   cudaMemcpyFromArrayAsync   ///
typedef cudaError_t (*cudaMemcpyFromArrayAsync_t)(void * dst,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpyFromArrayAsync_t native_cudaMemcpyFromArrayAsync = NULL;

extern "C" cudaError_t cudaMemcpyFromArrayAsync (void * dst,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream){

    printf("\n>>cudaMemcpyFromArrayAsync interception\n");

    if (native_cudaMemcpyFromArrayAsync == NULL) {
        native_cudaMemcpyFromArrayAsync = (cudaMemcpyFromArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyFromArrayAsync");
    }
    assert(native_cudaMemcpyFromArrayAsync != NULL);
    return native_cudaMemcpyFromArrayAsync(dst,src,wOffset,hOffset,count,kind,stream);
}


///   cudaMemcpyFromSymbol   ///
typedef cudaError_t (*cudaMemcpyFromSymbol_t)(void * dst,
    const char * symbol,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind);

static cudaMemcpyFromSymbol_t native_cudaMemcpyFromSymbol = NULL;

extern "C" cudaError_t cudaMemcpyFromSymbol (void * dst,
    const char * symbol,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind) {

    printf("\n>>cudaMemcpyFromSymbol interception\n");

    if (native_cudaMemcpyFromSymbol == NULL) {
        native_cudaMemcpyFromSymbol = (cudaMemcpyFromSymbol_t)dlsym(RTLD_NEXT,"cudaMemcpyFromSymbol");
    }
    assert(native_cudaMemcpyFromSymbol != NULL);
    return native_cudaMemcpyFromSymbol(dst,symbol,count,offset,kind);
}


///   cudaMemcpyFromSymbolAsync   ///
typedef cudaError_t (*cudaMemcpyFromSymbolAsync_t)(void * dst,
    const char * symbol,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpyFromSymbolAsync_t native_cudaMemcpyFromSymbolAsync = NULL;

extern "C" cudaError_t cudaMemcpyFromSymbolAsync (void * dst,
    const char * symbol,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {

    printf("\n>>cudaMemcpyFromSymbolAsync interception\n");

    if (native_cudaMemcpyFromSymbolAsync == NULL) {
        native_cudaMemcpyFromSymbolAsync = (cudaMemcpyFromSymbolAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyFromSymbolAsync");
    }
    assert(native_cudaMemcpyFromSymbolAsync != NULL);
    return native_cudaMemcpyFromSymbolAsync(dst,symbol,count,offset,kind,stream);
}


///   cudaMemcpyToArray   ///
typedef cudaError_t (*cudaMemcpyToArray_t)(struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind);

static cudaMemcpyToArray_t native_cudaMemcpyToArray = NULL;

extern "C" cudaError_t cudaMemcpyToArray (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind) {

    printf("\n>>cudaMemcpyToArray interception\n");

    if (native_cudaMemcpyToArray == NULL) {
        native_cudaMemcpyToArray = (cudaMemcpyToArray_t)dlsym(RTLD_NEXT,"cudaMemcpyToArray");
    }
    assert(native_cudaMemcpyToArray != NULL);
    return native_cudaMemcpyToArray(dst,wOffset,hOffset,src,count,kind);
}


///   cudaMemcpyToArrayAsync   ///
typedef cudaError_t (*cudaMemcpyToArrayAsync_t)(struct cudaArray * 	dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpyToArrayAsync_t native_cudaMemcpyToArrayAsync = NULL;

extern "C" cudaError_t cudaMemcpyToArrayAsync (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {

    printf("\n>>cudaMemcpyToArrayAsync interception\n");

    if (native_cudaMemcpyToArrayAsync == NULL) {
        native_cudaMemcpyToArrayAsync = (cudaMemcpyToArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyToArrayAsync");
    }
    assert(native_cudaMemcpyToArrayAsync != NULL);
    return native_cudaMemcpyToArrayAsync(dst,wOffset,hOffset,src,count,kind,stream);
}


///   cudaMemcpyToSymbol   ///
typedef cudaError_t (*cudaMemcpyToSymbol_t)(const char * symbol,
    const void * src,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind);

static cudaMemcpyToSymbol_t native_cudaMemcpyToSymbol = NULL;

extern "C" cudaError_t cudaMemcpyToSymbol (const char * symbol,
    const void * src,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind) {

    printf("\n>>cudaMemcpyToSymbol interception\n");

    if (native_cudaMemcpyToSymbol == NULL) {
        native_cudaMemcpyToSymbol = (cudaMemcpyToSymbol_t)dlsym(RTLD_NEXT,"cudaMemcpyToSymbol");
    }
    assert(native_cudaMemcpyToSymbol != NULL);
    return native_cudaMemcpyToSymbol(symbol,src,count,offset,kind);
}


///   cudaMemcpyToSymbolAsync   ///
typedef cudaError_t (*cudaMemcpyToSymbolAsync_t)(const char * symbol,
    const void * src,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);

static cudaMemcpyToSymbolAsync_t native_cudaMemcpyToSymbolAsync = NULL;

extern "C" cudaError_t cudaMemcpyToSymbolAsync (const char * symbol,
    const void * src,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {

    printf("\n>>cudaMemcpyToSymbolAsync interception\n");

    if (native_cudaMemcpyToSymbolAsync == NULL) {
        native_cudaMemcpyToSymbolAsync = (cudaMemcpyToSymbolAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyToSymbolAsync");
    }
    assert(native_cudaMemcpyToSymbolAsync != NULL);
    return native_cudaMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream);
}


///   cudaMemset   ///
typedef cudaError_t (*cudaMemset_t)(void * devPtr, int value, size_t count);
static cudaMemset_t native_cudaMemset = NULL;

extern "C" cudaError_t cudaMemset(void * devPtr, int value, size_t count) {
    printf("\n>>cudaMemset interception\n");

    if (native_cudaMemset == NULL) {
        native_cudaMemset = (cudaMemset_t)dlsym(RTLD_NEXT,"cudaMemset");
    }
    assert(native_cudaMemset != NULL);
    return native_cudaMemset(devPtr,value,count);
}


///   cudaMemset2D   ///
typedef cudaError_t (*cudaMemset2D_t)(void * devPtr,
    size_t  pitch,
    int     value,
    size_t 	width,
    size_t 	height);

static cudaMemset2D_t native_cudaMemset2D = NULL;

extern "C" cudaError_t cudaMemset2D (void * devPtr,
    size_t  pitch,
    int     value,
    size_t 	width,
    size_t 	height) {

    printf("\n>>cudaMemset2D interception\n");

    if (native_cudaMemset2D == NULL) {
        native_cudaMemset2D = (cudaMemset2D_t)dlsym(RTLD_NEXT,"cudaMemset2D");
    }
    assert(native_cudaMemset2D != NULL);
    return native_cudaMemset2D(devPtr,pitch,value,width,height);
}


///   cudaMemset3D   ///
typedef cudaError_t (*cudaMemset3D_t)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);

static cudaMemset3D_t native_cudaMemset3D = NULL;

extern "C" cudaError_t cudaMemset3D (struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
    printf("\n>>cudaMemset3D interception\n");

    if (native_cudaMemset3D == NULL) {
        native_cudaMemset3D = (cudaMemset3D_t)dlsym(RTLD_NEXT,"cudaMemset3D");
    }
    assert(native_cudaMemset3D != NULL);
    return native_cudaMemset3D(pitchedDevPtr,value,extent);
}



//***********************************************//
//      CUDA Runtime API Version Management      //
//***********************************************//
///   cudaDriverGetVersion   ///
typedef cudaError_t (*cudaDriverGetVersion_t)(int * driverVersion);
static cudaDriverGetVersion_t native_cudaDriverGetVersion = NULL;

extern "C" cudaError_t cudaDriverGetVersion	(int * driverVersion) {
    printf("\ncudaDriverGetVersion interception\n");

    if (native_cudaDriverGetVersion == NULL) {
        native_cudaDriverGetVersion = (cudaDriverGetVersion_t)dlsym(RTLD_NEXT,"cudaDriverGetVersion");
    }
    assert(native_cudaDriverGetVersion != NULL);
    return native_cudaDriverGetVersion(driverVersion);
}

///   cudaDriverGetVersion   ///
typedef cudaError_t (*cudaRuntimeGetVersion_t)(int * runtimeVersion);
static cudaRuntimeGetVersion_t native_cudaRuntimeGetVersion = NULL;

extern "C" cudaError_t cudaRuntimeGetVersion(int * runtimeVersion) {
    printf("\ncudaRuntimeGetVersion interception\n");

    if (native_cudaRuntimeGetVersion == NULL) {
        native_cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)dlsym(RTLD_NEXT,"cudaRuntimeGetVersion");
    }
    assert(native_cudaRuntimeGetVersion != NULL);
    return native_cudaRuntimeGetVersion(runtimeVersion);
}



//**********************************************//
//      CUDA Runtime API Thread Management      //
//**********************************************//
///   cudaThreadExit   ///
typedef cudaError_t (*cudaThreadExit_t)(void);
static cudaThreadExit_t native_cudaThreadExit = NULL;

extern "C" cudaError_t cudaThreadExit(void) {
    printf("\n>>cudaThreadExit interception\n");

    if (native_cudaThreadExit == NULL) {
        native_cudaThreadExit = (cudaThreadExit_t)dlsym(RTLD_NEXT,"cudaThreadExit");
    }
    assert(native_cudaThreadExit != NULL);
    return native_cudaThreadExit();
}

///   cudaThreadExit   ///
typedef cudaError_t (*cudaThreadSynchronize_t)(void);
static cudaThreadSynchronize_t native_cudaThreadSynchronize = NULL;

extern "C" cudaError_t cudaThreadSynchronize(void) {
    printf("\n>>cudaThreadSynchronize interception\n");

    if (native_cudaThreadSynchronize == NULL) {
        native_cudaThreadSynchronize = (cudaThreadSynchronize_t)dlsym(RTLD_NEXT,"cudaThreadSynchronize");
    }
    assert(native_cudaThreadSynchronize != NULL);
    return native_cudaThreadSynchronize();
}

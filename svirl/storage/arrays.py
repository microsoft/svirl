# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda

from warnings import warn

class GArray(object):
    """Returns a GArray object upon allocating memory on either
    host, device, or host and device. 

    Allocations on host and device are done using numpy and PyCUDA's
    gpuarray respectively. Allocations on multiple devices are not 
    supported yet.

    Assumes:
      o cuda context for this device has already been created.
      o GPUArrays are 1D while numpy arrays can be multi-dimensional.
         if multi-dimensional arrays are given as inputs, the memory
         allocation will correspond to their flattened version

    Parameters
    ----------
    nelements  : (required with shape) Accepts positive integers and specifies the number of
                 elements in the array. Required if argument `like`
                 is not specified (and ignored if specified with `like`)

    shape      : (optional) Accepts a tuple of integers that specifies the shape of
                 the array on host. The corresponding array on GPU, if 
                 requested, will be flattened. shape will be ignored if
                 specified with the parameter `like` since it will be 
                 derived from the prototype

                 Shape can be a list of tuples to indicate vector
                 storage. The corresponding array on GPU, if requested,
                 will be flattened in the same order as the list.

    dtype      : (Optional) Accepts only numpy float and complex datatypes. 
                 If not specified, np.float64 is assumed. 
                 Supported dtypes:
                  (np.float32, np.float64, np.complex64, np.complex128) 

    on         : (Optional) Accepts only one of the following class variables:
                  on_device,
                  on_host_and_device

    like       : (Optional) Accepts a numpy, pycuda's gpuarray, or a GArray object 
                 as a prototype and creates a new object inheriting the
                 properties of the prototype.

                 If prototype is GArray with allocations on both host 
                 and device, make sure the data is synced. In this case,
                 the data will be fetched from the device.

    like_init  : (Optional) Accepts a boolean. If this is set to True (default), 
                 the allocated object will be initialized to the 
                 prototype's value.

                 This property is not inherited if another GArray
                 object is passed using `like` object and will use 
                 the default value (GArray.on_host_and_device)

                 When this is set to True, the data is copied from 
                 host array, and only if is not available, data from 
                 device array is copied. If the prototype is a GArray 
                 and allocated on both host and device, make sure that
                 the data is synced before using this option.
                           
    name       : (Optional) Accepts a string as a name for the array. 
    


    Examples:
       1. Create a new array on host and device based on a host array, 
          and init the array (default)
             gabei = GArray(like = abei, on = GArray.on_device)
       2. Create a new array with size and dtype of the array as input
             psi   = GArray(N, dtype = np.complex128)
       3. Create a new array on host based on a device array, 
          and don't init the array
             a     = GArray(like = garray, like_init = False, 
                              on = GArray.on_host_and_device)


    """

    # Memory allocations can be done only on:
    on_device           = 'd'
    on_host_and_device  = 'hd'

    def __init__(self,
            nelements = None,
            dtype = np.float64,
            shape = None, 
            on = None,
            like = None,
            like_init = True, 
            name = 'array-0',
            suppress_warnings = False
            ):


        self.__suppress_warnings = suppress_warnings

        ## Sanity checks

        if on is not None:
            supported_ons = (self.on_device, self.on_host_and_device)
            if on not in supported_ons:
                raise ValueError("\n The value for `on` should be one of \
                        these: on_device, on_host_and_device")

        if like is not None:
            supported_protytypes = (np.ndarray, gpuarray.GPUArray, GArray)

            if not isinstance(like, tuple(supported_protytypes)):
                raise TypeError("\n `like` supports only the following \
                        prototypes: \n", supported_protytypes)

            if shape is not None:
                self.__warn("\n Shape argument will be ignored since \
                        a prototype is given.")

        if like is None:
            if nelements is None and shape is None:
                raise ValueError("\n Specify size using `nelements` \
                        options or specify the `shape` of the array")

            if nelements is not None and shape is not None:
                self.__warn("\n Both size and shape of the array \
                        specified. Using shape as the input.")
                nelements = None # Note that this will be updated later

        supported_types = (np.float32, np.float64, np.complex64, np.complex128) 
        if dtype not in supported_types:
            raise TypeError("\n `dtype` supports only the \
                    following types: \n", supported_types)

        # should we assert that if shape is a list, 
        # its elements should be a tuple?

        ## Fill attributes

        self._data  = None  # host array
        self._gdata = None  # device array
        self._data_v = None # pointers to host array for vector storage

        # NOTE: tested vector storage impl only for "host and device"

        self._nelements = None
        self._nelements_v = None

        # sync_status holds synchronization state between host and device:
        #    -1: Data from host need to be copied to device
        #     0: Data from device and host are synchronized
        #     1: Data from device need to be copied to host
        self.__sync_status = None

        # ndim = 1: scalar storage, 2: vector storage
        self._ndim = 1
        if shape is not None and isinstance(shape, list):
            self._ndim = len(shape)
            self._nelements_v = np.zeros(self._ndim, dtype=np.uint32)

        # 3D not supported yet
        assert self._ndim > 0 and self._ndim < 3

        self._shape = shape
        self._dtype = dtype
        self._like = like

        self.name = str(name)

        # Prototype given: Derive attributes from it
        if like is not None:
            if on is None:
                on = self.on_host_and_device

            self._nelements = like.size

            self._shape = like.shape
            self._dtype = like.dtype

        # No prototype: Fill attributes from input 
        if like is None:
            like_init = False

            if on is None:
                on = self.on_host_and_device

            # Shape provided: set nelements (_v) based on shape
            if shape is not None:
                self._shape = shape
                if self._ndim == 1:
                    self._nelements = np.prod(shape)
                else:
                    for idim in range(self._ndim):
                        self._nelements_v[idim] = np.prod(shape[idim])

                    self._nelements = np.sum(self._nelements_v)
                    self._nelements_v = np.cumsum(self._nelements_v)

            # nelements provided: set shape based on nelements
            if nelements is not None:
                assert self._ndim == 1
                self._shape = (nelements, 1)
                self._nelements = nelements

        self._on = on
        self._like_init = like_init

        if self.__on_host_and_device():
            self.synced()

        if self._shape is None:
            raise ValueError("Shape is none")

        if self._nelements is None:
            raise ValueError("nelements is none")

        # Allocate memory on host and/or device
        self.__alloc_storage()

        # TODO; for now assume we have the data on host
        if self._ndim >1:
            self._data_v = np.split(self._data, self._nelements_v) 

        # Unsave the reference to prototype
        self._like = None


    def __del__(self):
        if self.__on_device() and self._gdata is not None:
            self._gdata.gpudata.free()


    ## Exposed Methods 

    @property
    def size(self):
        """
        Returns the size of the array. For multi-dimensional arrays,
        it is the product of sizes in each dimension.
        """
        return self._nelements


    @property
    def dtype(self):
        """
        Returns the datatype (dtype) of the elements in the array.
        """
        return self._dtype


    @property
    def shape(self):
        """
        Returns the shape of the array as a tuple of integers.
        """
        return self._shape


    @property
    def on(self):
        """
        Returns where the allocations for the object have been made
        and is one of the following:

        on_device ('d'),
        on_host_and_device ('hd')
        """
        return self._on


    def sync(self):
        """
        Sync the data between host and device if needed.
        Remember to update the sync status if the data is changed. 
        This can be done using the following instance methods:

        need_htod_sync(),
        need_dtoh_sync(),
        synced()

        """
        if not self.__on_host_and_device():
            return

        self.__sync()


    def need_dtoh_sync(self):
        """
        Invoke this method to indicate that the data from
        device need to be synced with data at host. 
        It does not sync, only indicates that a sync is needed
        """
        if self.__on_host_and_device():
            self.__sync_status = 1


    def need_htod_sync(self):
        """
        Invoke this method to indicate that the data from
        host need to be synced with data at device. 
        It does not sync, only indicates that a sync is needed
        """
        if self.__on_host_and_device():
            self.__sync_status = -1


    def synced(self):
        """
        Invoke this method to indicate that the data at 
        host and device is synced.
        """
        if self.__on_host_and_device():
            self.__sync_status = 0


    # Array operations:

    def __flatten_array_with_size(self, arr, size):
        if arr is not None:
            return np.reshape(arr.T, size)

        return None


    def __flatten_array(self, arr = None):
        """
        If an input array is given, flatten it to the current size of the
        array. If not given, return the flattened data on the host
        if available, else return None
        """
        if arr is None:
            return self.__flatten_array_with_size(self._data, self._nelements)
        else:
            return self.__flatten_array_with_size(arr, self._nelements)


    def __unflatten_array_with_shape(self, arr, shape):
        if arr is not None:
            return np.reshape(arr, tuple(reversed(shape))).T

        return None


    def __unflatten_array(self, use_device_data = False):
        """
        Unflatten the device/host data to the shape of the array
        corresponding to this object.

        If `use_device_data` is True, device data is fetched and 
        unflattened. Else, host data is unflattened if available, 
        otherwise None is returned.
        """
        if use_device_data:
            return self.__unflatten_array_with_shape(self._gdata.get(), self._shape)
        else:
            return self.__unflatten_array_with_shape(self._data, self._shape)


    # Getters

    def get_h(self, sync = True):
        """
        Return the host array (not a copy)
        """
        if sync is True:
            self.sync()

        # Need to define appropriate behavior
        if self._ndim > 1:
            return None

        return self.__unflatten_array()


    def get_d(self, sync = True):
        """
        Return the device array (not a copy)
        """
        if sync is True:
            self.sync()

        # Need to define appropriate behavior
        if self._ndim > 1:
            return None

        return self.__unflatten_array(use_device_data = True)


    def get_d_obj(self, sync = False):
        """
        Return the device array object. Sync is not performed by default.
        """
        if sync is True:
            self.sync()

        return self._gdata


    def get_vec_h(self, sync = True):
        """
        Return the components of vector storage on host unflattend
        """

        if sync is True:
            self.sync()

        if self._ndim > 1:
            a = self.__unflatten_array_with_shape(self._data_v[0], self._shape[0])
            b = self.__unflatten_array_with_shape(self._data_v[1], self._shape[1])
            return (a, b)

        return (None, None)

    # Setters

    def set_h(self, arr):
        """
        Copy the contents of `arr` to the host array and update
        the host to device sync status. Multi-dimensional
        arrays are flattened before copying.
        
        """
        if arr.size != self._nelements:
            warn("Size mismatch: Storage has size %d but input array has: %d" % (self._nelements, arr.size))
        else:
            np.copyto(self._data, self.__flatten_array(arr))

            self.need_htod_sync()


    def set_vec_h(self, arr_a, arr_b):
        """
        Set components of the vector storage 
        """
        assert arr_a.shape == self._shape[0]
        assert arr_b.shape == self._shape[1]

        if self._data is not None:
            data_a = self.__flatten_array_with_size(arr_a, arr_a.size)
            np.copyto(self._data_v[0], data_a)

            data_b = self.__flatten_array_with_size(arr_b, arr_b.size)
            np.copyto(self._data_v[1], data_b)

            self.need_htod_sync()


    # Memory Alloc/free
            
    def free(self):
        """
        Free the host and device memory on demand
        
        """
        if self._gdata is not None:
            self._gdata.gpudata.free()
            self._gdata = None

        if self._data is not None:
            del self._data

        if self._data_v is not None:
            del self._data_v

        self._nelements = 0
        self._shape = None
        self._dtype = None

            
    # Diagnostics/Utils

    def metadata(self):
        """
        Print object metadata like name, size, shape, on etc 
        
        """
        
        print('  Name       : ', self.name, flush = True)
        print('  size       : ', self._nelements, flush = True)
        print('  shape      : ', self._shape, flush = True)
        print('  dtype      : ', self._dtype, flush = True)
        print('  on         : ', self._on, flush = True)
        print('  sync status: ', self.__sync_status, flush = True)
        print('', flush = True)


    ### Private members


    def __on_device(self):
        return (self._on == self.on_device 
                or self._on == self.on_host_and_device)


    def __on_host_and_device(self):
        return (self._on == self.on_host_and_device)


    def __is_prototype_GArray(self):
        return isinstance(self._like, GArray)


    def __is_prototype_ndarray(self):
        return isinstance(self._like, np.ndarray)


    def __is_prototype_GPUArray(self):
        return isinstance(self._like, gpuarray.GPUArray)


    #  ------------------------------------------
    # | Prototype  | Prototype On | Requested On |
    #  ------------------------------------------
    #   GArray       HD, D           HD, D  <== Prototype host/device 
    #   ndarray        N/A           HD, D      data should be synced     
    #   GPUArray       N/A           HD, D
    #  ------------------------------------------


    # Allocations similar to the specified prototype + initialization
    def __alloc_like_init(self):

        # If prototype is GArray, use get_d_obj() to access the data 
        def alloc_from_GArray():
            self._gdata = self._like.get_d_obj().copy()
            if self.__on_host_and_device():
                self._data = self._gdata.get()
                self.synced()

        def alloc_from_ndarray():
            if self.__on_host_and_device():
                self._data = self.__flatten_array(self._like).copy()
                self._gdata = gpuarray.empty(self._like.size, dtype = self._dtype)
                cuda.memcpy_htod(self._gdata.gpudata, self._data)
                self.synced()
            elif self.__on_device():
                self._gdata = gpuarray.empty(self._like.size, dtype = self._dtype)
                flattened_array = self.__flatten_array(self._like)
                cuda.memcpy_htod(self._gdata.gpudata, flattened_array)

        def alloc_from_GPUArray():
            self._gdata = self._like.copy()
            if self.__on_host_and_device():
                self._data = self._gdata.get()
                self.synced()


        if self.__is_prototype_GArray():
            alloc_from_GArray()
        elif self.__is_prototype_ndarray():
            alloc_from_ndarray()
        elif self.__is_prototype_GPUArray():
            alloc_from_GPUArray()
        else:
            warn('Unsupported prototype!')


    def __alloc_default(self):
        if self.__on_device():
            self._gdata = gpuarray.empty(self._nelements, dtype = self._dtype)

        if self.__on_host_and_device(): 
            self._data = np.zeros(self._nelements, dtype = self._dtype)
            self._gdata = gpuarray.empty(self._nelements, dtype = self._dtype)
            self.synced()


    def __alloc_storage(self):
        if self._like_init is True:
            self.__alloc_like_init()
        else:
            self.__alloc_default()

    
    def __sync(self):
        if self.__need_dtoh_sync():       
            cuda.memcpy_dtoh(self._data, self._gdata.gpudata)
        elif self.__need_htod_sync(): 
            cuda.memcpy_htod(self._gdata.gpudata, self._data)

        self.synced()


    def __need_htod_sync(self):
        return (self.__sync_status is not None and self.__sync_status < 0)


    def __need_dtoh_sync(self):
        return (self.__sync_status is not None and self.__sync_status > 0)


    # Utils

    def __warn(self, msg):
        if not isinstance(msg, str):
            return

        if self.__suppress_warnings is False:
            warn(msg)


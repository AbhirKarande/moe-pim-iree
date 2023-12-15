// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <fstream>

// library for PIM SDK
#include <algorithm>
#include "pim.h"
#include <unistd.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/hal/drivers/vulkan/PIM_buffer.h"

// json header
#include "json/json.h"
#include "iostream"
#include <string>
#include <ctime>

IREE_FLAG(string, function, "",
          "Name of a function contained in the module specified by --module= "
          "to run.");
  
IREE_FLAG(string, meta_data, "",
          "meta_data of the module. It contains the information of dimension and data parallelize to each device");



namespace iree {
namespace {

iree_status_t Run(int* out_exit_code) {
  IREE_TRACE_SCOPE0("iree-run-module");

  iree_allocator_t host_allocator = iree_allocator_system();
  vm::ref<iree_vm_instance_t> instance;
  IREE_RETURN_IF_ERROR(iree_tooling_create_instance(host_allocator, &instance),
                       "creating instance");

  vm::ref<iree_vm_module_t> main_module;
  IREE_RETURN_IF_ERROR(iree_tooling_load_module_from_flags(
      instance.get(), host_allocator, &main_module));

  vm::ref<iree_vm_context_t> context;
  vm::ref<iree_hal_device_t> device;
  vm::ref<iree_hal_allocator_t> device_allocator;
  IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
      instance.get(), /*user_module_count=*/1, /*user_modules=*/&main_module,
      /*default_device_uri=*/iree_string_view_empty(), host_allocator, &context,
      &device, &device_allocator));

  std::string function_name = std::string(FLAG_function);
  iree_vm_function_t function;
  if (function_name.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no --function= specified");
  } else {
    IREE_RETURN_IF_ERROR(
        iree_vm_module_lookup_function_by_name(
            main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
            iree_string_view_t{function_name.data(), function_name.size()},
            &function),
        "looking up function '%s'", function_name.c_str());
  }

  IREE_RETURN_IF_ERROR(iree_hal_begin_profiling_from_flags(device.get()));

  // meta data exception
  Json::Value root;

  std::ifstream meta_file(FLAG_meta_data, std::ios::in);

  if (true != meta_file.is_open()){
    printf("\n[IREE-PiM] meta_file open failed!!\n");
    return iree_ok_status();
  }

  meta_file >> root;
  
  meta_file.close();

  print_pim_SDK(0);

  // shape 
  std::vector<std::vector<int>> meta_shape;
  std::vector<int> meta_rank;

  ///////////////////////////////////////////////////////////////////////////////////////
  //                                                                                   //
  //   meta_shape        :  shape of tensor                                            //   
  //                                                                                   //  
  //   meta_rank         :  rank of tensor                                             //
  //                                                                                   //
  ///////////////////////////////////////////////////////////////////////////////////////

  std::cout << std::endl << "[IREE-PiM] Decode meta_data.json" << std::endl << std::endl;

  int n_of_params = root.size()-1;

  for(int i=0; i<root.size()-1; i++){
    std::string name = std::to_string(i+1);

    // get shape from meta data
    std::vector<int> tmp_pair;
    
    if (root[name]["rank"].asInt()==3 ){
      tmp_pair.push_back(root[name]["shape"][0].asInt());
      tmp_pair.push_back(root[name]["shape"][1].asInt());
      tmp_pair.push_back(root[name]["shape"][2].asInt());
      meta_shape.push_back(tmp_pair);
      meta_rank.push_back(3);
    }
    else if (root[name]["rank"].asInt()==2 ){
      tmp_pair.push_back(root[name]["shape"][0].asInt());
      tmp_pair.push_back(root[name]["shape"][1].asInt());
      meta_shape.push_back(tmp_pair);
      meta_rank.push_back(2);
    }  
  }

  //////////// data processing ///////////
  std::vector<iree_hal_buffer_view_t*> params_vec;  // [# of parameter][# of layer]

  for (int j=0; j<n_of_params; j++){
    params_vec.push_back(nullptr);
  }   
    
  // parameter generation  &  make hal_buffer to reuse parameter
  // generate meta_data defined random parameters
  for (int i =0; i<n_of_params; i++){

    int data_size=0;
    int shape_0=0;
    int shape_1=0;
    int shape_2=0;

    // reshape to PiM hardware dimension layout
    if (meta_rank[i]==2){
      data_size = meta_shape[i][0] * meta_shape[i][1];
      shape_0=meta_shape[i][0];
      shape_1=meta_shape[i][1];
      shape_2=0;
    }
    else if (meta_rank[i]==3){
      data_size = meta_shape[i][0] * meta_shape[i][1] * meta_shape[i][2];
      shape_0=meta_shape[i][0];
      shape_1=meta_shape[i][1];
      shape_2=meta_shape[i][2];
    }
    
    // random tensor initialize
    std::vector<float> data(data_size, 0.0f);

    srand((unsigned int)(time(NULL)+i));
    
    for(int j=0; j<data_size; j++){
      int rand_data_tmp = rand()%200;
      float rand_data = (float)(rand_data_tmp-99) / 100;

      data[j] = rand_data;
    }
    
    params_vec[i] = iree_tooling_pim_alloc_data(
              device_allocator.get(),
              data.data(), shape_0, shape_1, shape_2, meta_rank[i]
              );
  }

  printf("\n\n[IREE-PiM]  Parameter generation done!!! \n\n");  


  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr, 16, host_allocator, &outputs));

  vm::ref<iree_vm_list_t> inputs_test;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(nullptr, n_of_params, host_allocator, &inputs_test));
  
  // perform mac with same weight
  // update history and push back to IREE input_list  
  for (int k=0; k<root.size()-1; k++){
    iree_tooling_pim_input_push(params_vec[k], &inputs_test);
  }

  printf("\n\n\n[IREE-PiM]  Compiler Module Invocation Begin   \n");   

  printf("\n\nEXEC @%s\n", function_name.c_str());
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context.get(), function, IREE_VM_INVOCATION_FLAG_NONE,
                  /*policy=*/nullptr, inputs_test.get(), outputs.get(),
                  host_allocator),
      "invoking function '%s'", function_name.c_str());
        

  printf("\n\n\n[IREE-PiM]  iterative invocation done   \n\n\n");

  IREE_RETURN_IF_ERROR(iree_hal_end_profiling_from_flags(device.get()));

  IREE_RETURN_IF_ERROR(iree_pim_read_buffer_view(outputs.get()));

  device_allocator.reset();
  device.reset();
  instance.reset();

  printf("\n\n\n[IREE-PiM]  execution done              \n\n\n");

  return iree_ok_status();
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc > 1) {
    // Avoid iree-run-module spinning endlessly on stdin if the user uses single
    // dashes for flags.
    printf(
        "[ERROR] unexpected positional argument (expected none)."
        " Did you use pass a flag with a single dash ('-')?"
        " Use '--' instead.\n");
    return 1;
  }

  int exit_code = EXIT_SUCCESS;
  iree_status_t status = Run(&exit_code);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }

  return exit_code;
}

}  // namespace iree

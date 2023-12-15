func.func @gpt2block(%input: tensor<1x768xf32>, %w_c_attn: tensor<768x2304xf32>,  %b_c_attn: tensor<1x2304xf32>, 
    %w_key_head: tensor<12x64x1024xf32>, %w_val_head: tensor<12x1024x64xf32>, %w_c_proj: tensor<768x768xf32>, %b_c_proj: tensor<1x768xf32>,
    %w_fc1: tensor<768x3072xf32>, %b_fc1: tensor<1x3072xf32>, %w_fc2: tensor<3072x768xf32>, %b_fc2: tensor<1x768xf32>) -> tensor<1x768xf32> {

    %init = stablehlo.constant dense<0x00000000> : tensor<f32>
    %eps = stablehlo.constant dense<0.000000000001> : tensor<1xf32> // Epsilon for layernorm

    // hist_ = 1024, #_of_head = 12

    ////////////////////////////////////////////////////////////////////////////////////////
    // LayerNorm 1
    ////////////////////////////////////////////////////////////////////////////////////////
    
    %ln1_sum = stablehlo.reduce(%input init: %init) across dimensions = [1] : (tensor<1x768xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %tmp = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %tmp : tensor<f32>
    }
    %ln1_sum_brd = stablehlo.broadcast_in_dim %ln1_sum, dims = [0] : (tensor<1xf32>) -> tensor<1x768xf32>
    %ln1_e = stablehlo.divide %input, %ln1_sum_brd : tensor<1x768xf32>
    %ln1_e_sub = stablehlo.subtract %input, %ln1_e : tensor<1x768xf32>
    %ln1_e_sub_sq = stablehlo.multiply %ln1_e_sub, %ln1_e_sub : tensor<1x768xf32>
    %ln1_e_sub_sq_sum = stablehlo.reduce(%ln1_e_sub_sq init: %init) across dimensions = [1] : (tensor<1x768xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %tmp = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %tmp : tensor<f32>
    }
    %ln1_e_sub_sq_sum_brd = stablehlo.broadcast_in_dim %ln1_e_sub_sq_sum, dims = [0] : (tensor<1xf32>) -> tensor<1x768xf32>
    
    %ln1_var = stablehlo.divide %input, %ln1_e_sub_sq_sum_brd : tensor<1x768xf32>
    %ln1_eps_brd = stablehlo.broadcast_in_dim %eps, dims = [0] : (tensor<1xf32>) -> tensor<1x768xf32>
    %ln1_eps_add = stablehlo.add %ln1_var, %ln1_eps_brd : tensor<1x768xf32>
    %ln1_rsqrt = stablehlo.rsqrt %ln1_eps_add : tensor<1x768xf32>
    %ln1 = stablehlo.multiply %ln1_e_sub, %ln1_rsqrt : tensor<1x768xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // c_attn
    ////////////////////////////////////////////////////////////////////////////////////////

    // c_attn : Q, K, V generation
    %c_attn_mul = stablehlo.dot %ln1, %w_c_attn : (tensor<1x768xf32>, tensor<768x2304xf32>) -> tensor<1x2304xf32>
    %c_attn = stablehlo.add %c_attn_mul, %b_c_attn : (tensor<1x2304xf32>, tensor<1x2304xf32>) -> tensor<1x2304xf32>

    // Return Q, K, V
    %query = "stablehlo.slice" (%c_attn) {
        start_indices = dense<[0, 0]> : tensor<2xi64>,
        limit_indices = dense<[1, 768]> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } : (tensor<1x2304xf32>) -> tensor<1x768xf32>

    %key = "stablehlo.slice" (%c_attn) {
        start_indices = dense<[0, 768]> : tensor<2xi64>,
        limit_indices = dense<[1, 1536]> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } : (tensor<1x2304xf32>) -> tensor<1x768xf32>


    %value = "stablehlo.slice" (%c_attn) {
        start_indices = dense<[0, 1536]> : tensor<2xi64>,
        limit_indices = dense<[1, 2304]> : tensor<2xi64>,
        strides = dense<1> : tensor<2xi64>
    } : (tensor<1x2304xf32>) -> tensor<1x768xf32>


    ////////////////////////////////////////////////////////////////////////////////////////
    // Split head
    ////////////////////////////////////////////////////////////////////////////////////////

    %query_head = stablehlo.reshape %query : (tensor<1x768xf32>) -> tensor<12x1x64xf32>
    %key_head = stablehlo.reshape %key : (tensor<1x768xf32>) -> tensor<12x64x1xf32>
    %val_head = stablehlo.reshape %value : (tensor<1x768xf32>) -> tensor<12x1x64xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // Query * Key
    ////////////////////////////////////////////////////////////////////////////////////////

    // Query * Key
    %attn_weight = "stablehlo.dot_general"(%query_head, %w_key_head) {
        dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]
        >,
         precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<12x1x64xf32>, tensor<12x64x1024xf32>) -> tensor<12x1x1024xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // Softmax
    ////////////////////////////////////////////////////////////////////////////////////////
    
    // Softmax
    %5 = stablehlo.reduce(%attn_weight init: %init) across dimensions = [2] : (tensor<12x1x1024xf32>, tensor<f32>) -> tensor<12x1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %tmp = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %tmp : tensor<f32>
    }
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<12x1xf32>) -> tensor<12x1x1024xf32>
    %7 = stablehlo.subtract %attn_weight, %6 : tensor<12x1x1024xf32>
    %8 = stablehlo.exponential %7 : tensor<12x1x1024xf32>
    %9 = stablehlo.reduce(%8 init: %init) across dimensions = [2] : (tensor<12x1x1024xf32>, tensor<f32>) -> tensor<12x1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %tmp = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %tmp : tensor<f32>
    }
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<12x1xf32>) -> tensor<12x1x1024xf32>
    %attn_score = stablehlo.divide %8, %10 : tensor<12x1x1024xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // Score * Value
    ////////////////////////////////////////////////////////////////////////////////////////

    %attn_output_head = "stablehlo.dot_general"(%attn_score, %w_val_head) {
        dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]
        >,
         precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<12x1x1024xf32>, tensor<12x1024x64xf32>) -> tensor<12x1x64xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // Merge heads
    ////////////////////////////////////////////////////////////////////////////////////////

    %attn_output = stablehlo.reshape %attn_output_head : (tensor<12x1x64xf32>) -> tensor<1x768xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // c_proj
    ////////////////////////////////////////////////////////////////////////////////////////

    // Stage4 : FC layer of Output
    %c_proj_mul = stablehlo.dot %attn_output, %w_c_proj : (tensor<1x768xf32>, tensor<768x768xf32>) -> tensor<1x768xf32>
    %c_proj = stablehlo.add %c_proj_mul, %b_c_proj : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // Shortcut 1
    ////////////////////////////////////////////////////////////////////////////////////////

    // Shortcut 1 
    %res1 = stablehlo.add %c_proj, %input : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // LayerNorm 2
    ////////////////////////////////////////////////////////////////////////////////////////

    %ln2_sum = stablehlo.reduce(%res1 init: %init) across dimensions = [1] : (tensor<1x768xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %tmp = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %tmp : tensor<f32>
    }
    %ln2_sum_brd = stablehlo.broadcast_in_dim %ln2_sum, dims = [0] : (tensor<1xf32>) -> tensor<1x768xf32>
    %ln2_e = stablehlo.divide %res1, %ln2_sum_brd : tensor<1x768xf32>
    %ln2_e_sub = stablehlo.subtract %res1, %ln2_e : tensor<1x768xf32>
    %ln2_e_sub_sq = stablehlo.multiply %ln2_e_sub, %ln2_e_sub : tensor<1x768xf32>
    %ln2_e_sub_sq_sum = stablehlo.reduce(%ln2_e_sub_sq init: %init) across dimensions = [1] : (tensor<1x768xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %tmp = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %tmp : tensor<f32>
    }
    %ln2_e_sub_sq_sum_brd = stablehlo.broadcast_in_dim %ln2_e_sub_sq_sum, dims = [0] : (tensor<1xf32>) -> tensor<1x768xf32>
    %ln2_var = stablehlo.divide %res1, %ln2_e_sub_sq_sum_brd : tensor<1x768xf32>
    %ln2_eps_brd = stablehlo.broadcast_in_dim %eps, dims = [0] : (tensor<1xf32>) -> tensor<1x768xf32>
    %ln2_eps_add = stablehlo.add %ln2_var, %ln2_eps_brd : tensor<1x768xf32>
    %ln2_rsqrt = stablehlo.rsqrt %ln2_eps_add : tensor<1x768xf32>
    %ln2 = stablehlo.multiply %ln2_e_sub, %ln2_rsqrt : tensor<1x768xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // FC1 (768x3072) + GELU
    ////////////////////////////////////////////////////////////////////////////////////////

    %fc1_mul = stablehlo.dot %ln2, %w_fc1 : (tensor<1x768xf32>, tensor<768x3072xf32>) -> tensor<1x3072xf32>
    %fc1 = stablehlo.add %fc1_mul, %b_fc1 : (tensor<1x3072xf32>, tensor<1x3072xf32>) -> tensor<1x3072xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // FC2 (3072x768)
    ////////////////////////////////////////////////////////////////////////////////////////

    %fc2_mul = stablehlo.dot %fc1, %w_fc2 : (tensor<1x3072xf32>, tensor<3072x768xf32>) -> tensor<1x768xf32>
    %fc2 = stablehlo.add %fc2_mul, %b_fc2 : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>

    ////////////////////////////////////////////////////////////////////////////////////////
    // Shortcut 2
    ////////////////////////////////////////////////////////////////////////////////////////

    %res2 = stablehlo.add %fc2, %res1 : (tensor<1x768xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>

    return %res2 : tensor<1x768xf32>

}
